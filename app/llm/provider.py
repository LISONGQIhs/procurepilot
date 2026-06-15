from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.client import IncompleteRead
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class LLMIntentResult(BaseModel):
    intent: str = Field(pattern="^(procurement|chitchat|out_of_scope|compliance_risk)$")
    reason: str


class LLMPurchaseExtraction(BaseModel):
    department: str | None = None
    purchase_category: str | None = None
    item_name: str | None = None
    quantity: int | None = None
    amount: float | None = None
    purpose: str | None = None
    vendor_name: str | None = None
    delivery_requirement: str | None = None
    budget_category: str | None = None
    budget_category_confirmed: bool = False
    is_urgent: bool = False
    specified_brand_or_model: bool = False
    flags: list[str] = Field(default_factory=list)


@dataclass
class LLMSettings:
    enabled: bool
    api_key: str | None
    base_url: str
    model: str
    timeout_seconds: float = 20.0

    @classmethod
    def from_env(cls) -> "LLMSettings":
        enabled = os.getenv("PROCUREPILOT_LLM_ENABLED", "false").strip().lower() == "true"
        return cls(
            enabled=enabled,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )


@dataclass
class LLMCallRecord:
    purpose: str
    model: str
    enabled: bool
    success: bool
    elapsed_ms: int
    fallback_used: bool = False
    output_summary: dict[str, Any] = field(default_factory=dict)
    token_usage: dict[str, Any] | None = None
    error_message: str | None = None


class LLMProvider:
    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or LLMSettings.from_env()

    @property
    def enabled(self) -> bool:
        return self.settings.enabled and bool(self.settings.api_key)

    def classify_intent(self, message: str) -> tuple[LLMIntentResult | None, LLMCallRecord]:
        system = (
            "你是企业采购合规预审智能体的意图分类器。"
            "只输出 JSON，不要输出解释文本。"
            "intent 必须是 procurement、chitchat、out_of_scope、compliance_risk 之一。"
            "compliance_risk 表示用户请求拆单、规避审批、绕过制度或其他采购合规违规协助。"
        )
        user = {
            "message": message,
            "schema": {"intent": "procurement|chitchat|out_of_scope|compliance_risk", "reason": "简短原因"},
        }
        result, record = self._json_chat("intent_classification", system, json.dumps(user, ensure_ascii=False))
        if not result:
            return None, record
        try:
            return LLMIntentResult.model_validate(result), record
        except ValidationError as exc:
            record.success = False
            record.error_message = f"intent validation failed: {exc}"
            record.fallback_used = True
            return None, record

    def extract_purchase_request(self, message: str) -> tuple[LLMPurchaseExtraction | None, LLMCallRecord]:
        system = (
            "你是企业采购合规预审智能体的字段抽取器。"
            "从中文采购需求中抽取结构化字段，只输出 JSON。"
            "不要编造用户未提供的信息；无法确认时用 null 或 false。"
            "金额统一输出人民币元数字；数量输出整数。"
            "specified_brand_or_model 必须输出布尔值 true 或 false，不要输出品牌名或型号名。"
            "flags 可包含 USER_COMPLIANCE_RISK、QUANTITY_AMBIGUOUS、DEPARTMENT_AMBIGUOUS、POSSIBLE_AMOUNT_CONFLICT。"
            "只有用户明确请求拆单、拆分合同、规避审批、绕过制度、不走审批等违规协助时，才输出 USER_COMPLIANCE_RISK；"
            "普通指定供应商、指定品牌或指定型号不是 USER_COMPLIANCE_RISK，应抽取为 vendor_name 或 specified_brand_or_model。"
        )
        user = {
            "message": message,
            "fields": [
                "department",
                "purchase_category",
                "item_name",
                "quantity",
                "amount",
                "purpose",
                "vendor_name",
                "delivery_requirement",
                "budget_category",
                "budget_category_confirmed",
                "is_urgent",
                "specified_brand_or_model",
                "flags",
            ],
        }
        result, record = self._json_chat("purchase_extraction", system, json.dumps(user, ensure_ascii=False))
        if not result:
            return None, record
        try:
            return LLMPurchaseExtraction.model_validate(result), record
        except ValidationError as exc:
            record.success = False
            record.error_message = f"purchase extraction validation failed: {exc}"
            record.fallback_used = True
            return None, record

    def polish_recommendation(self, context: dict[str, Any]) -> tuple[str | None, LLMCallRecord]:
        system = (
            "你是企业采购合规预审智能体的结果表达助手。"
            "只能基于输入的结构化结论润色 summary。"
            "不得改变 recommendation_type、risk_level、human_review_required、制度引用、工具结果或风险判断。"
            "只输出 JSON：{\"summary\":\"...\"}。"
        )
        result, record = self._json_chat("recommendation_polish", system, json.dumps(context, ensure_ascii=False))
        summary = result.get("summary") if result else None
        if not isinstance(summary, str) or not summary.strip():
            record.success = False
            record.fallback_used = True
            record.error_message = record.error_message or "missing summary"
            return None, record
        return summary.strip(), record

    def generate_boundary_reply(self, message: str, intent: str, reason: str) -> tuple[str | None, LLMCallRecord]:
        system = (
            "你是企业采购合规预审智能体 ProcurePilot 的边界回复助手。"
            "用户输入已被识别为闲聊或非采购请求。"
            "可以用简短自然的语气回应用户，但必须引导用户回到企业采购合规预审任务。"
            "不要处理无关任务本身，不要写诗、写代码、生成合同或完成其他非采购任务。"
            "不要生成采购审批结论，不要编造制度依据、预算、供应商、价格或审批链结果。"
            "不要声称已调用工具、已完成预审、已查询预算、已查询供应商或已完成审批。"
            "只输出 JSON：{\"reply\":\"...\"}。"
        )
        user = {
            "message": message,
            "intent": intent,
            "reason": reason,
            "reply_requirements": [
                "简短回应用户",
                "明确 ProcurePilot 只处理企业采购合规预审",
                "引导用户输入采购部门、品类、金额、用途、供应商等采购需求",
                "不承诺处理无关任务",
                "不输出采购结论、制度引用、预算、供应商、价格或审批结果",
            ],
        }
        result, record = self._json_chat("boundary_reply", system, json.dumps(user, ensure_ascii=False))
        reply = result.get("reply") if result else None
        if not isinstance(reply, str) or not reply.strip():
            record.success = False
            record.fallback_used = True
            record.error_message = record.error_message or "missing reply"
            return None, record
        return reply.strip(), record

    def _json_chat(self, purpose: str, system: str, user: str) -> tuple[dict[str, Any] | None, LLMCallRecord]:
        started = time.perf_counter()
        record = LLMCallRecord(
            purpose=purpose,
            model=self.settings.model,
            enabled=self.enabled,
            success=False,
            elapsed_ms=0,
        )
        if not self.enabled:
            record.fallback_used = True
            record.error_message = "LLM disabled or OPENAI_API_KEY missing"
            return None, record

        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        try:
            response_payload = self._post_chat_completion(payload)
            content = response_payload["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            record.success = True
            record.token_usage = response_payload.get("usage")
            record.output_summary = summarize_output(parsed)
            return parsed, record
        except urllib.error.HTTPError as exc:
            record.error_message = f"HTTP {exc.code}: {safe_read_error_body(exc)}"
            record.fallback_used = True
            return None, record
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError, OSError, IncompleteRead) as exc:
            record.error_message = str(exc)
            record.fallback_used = True
            return None, record
        finally:
            record.elapsed_ms = int((time.perf_counter() - started) * 1000)

    def _post_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for url in self._candidate_chat_urls():
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.settings.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
                    raw = response.read().decode("utf-8", errors="replace")
                return parse_chat_response(raw)
            except urllib.error.HTTPError:
                raise
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, IncompleteRead) as exc:
                last_error = exc
                continue
        if last_error:
            raise last_error
        raise RuntimeError("No chat completion endpoint configured")

    def _candidate_chat_urls(self) -> list[str]:
        base = self.settings.base_url.rstrip("/")
        urls = [f"{base}/chat/completions"]
        if not base.endswith("/v1"):
            urls.append(f"{base}/v1/chat/completions")
        return list(dict.fromkeys(urls))


def summarize_output(value: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, str):
            summary[key] = item[:120]
        elif isinstance(item, (int, float, bool)) or item is None:
            summary[key] = item
        elif isinstance(item, list):
            summary[key] = item[:6]
    return summary


def parse_chat_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("{"):
        return json.loads(text)

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        return json.loads(data)

    return json.loads(text)


def safe_read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read(500).decode("utf-8", errors="replace")
    except Exception:
        body = ""
    return body[:300]
