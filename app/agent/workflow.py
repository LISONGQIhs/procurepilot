from __future__ import annotations

from typing import Any

from app.agent.extractor import extract_purchase_request
from app.agent.hybrid_extractor import HybridExtractor
from app.agent.rag import PolicyRetriever
from app.llm.provider import LLMCallRecord, LLMIntentResult, LLMProvider
from app.models import (
    AgentRun,
    AgentState,
    AgentTraceEvent,
    ApprovalRecommendation,
    HumanReview,
    HumanReviewRequest,
    MissingInfoQuestion,
    PurchaseRequest,
    RiskFinding,
    ToolCallRecord,
    utc_now,
)
from app.services.store import RunStore
from app.tools.business_tools import BusinessTools


RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "BLOCKED": 3}
BOUNDARY_FALLBACK_SUMMARY = "ProcurePilot 当前只处理企业采购合规预审，请输入采购部门、品类、金额、用途等采购需求。"


class ProcurePilotAgent:
    def __init__(self, store: RunStore) -> None:
        self.store = store
        self.retriever = PolicyRetriever()
        self.tools = BusinessTools()
        self.llm_provider = LLMProvider()
        self.extractor = HybridExtractor(self.llm_provider)

    def precheck(self, message: str, run_id: str | None = None) -> AgentRun:
        if run_id:
            run = self.store.get(run_id)
            if run is None:
                raise KeyError(f"Run {run_id} not found")
            run.original_input = f"{run.original_input}\n用户补充：{message}"
            self._reset_for_resume(run)
        else:
            run = self.store.create(message)

        try:
            self._set_state(run, "INPUT_RECEIVED")
            self._trace(
                run,
                "USER_INPUT_RECEIVED",
                "收到采购预审输入",
                "创建或恢复一次采购合规预审运行。",
                {"message": message},
                {"run_id": run.run_id},
            )

            intent = self._classify_intent(run, run.original_input)
            if intent and intent.intent in {"chitchat", "out_of_scope"}:
                run.purchase_request = extract_purchase_request(run.original_input)
                self._set_state(run, "COMPLETED")
                summary = self._boundary_summary(run, run.original_input, intent)
                run.recommendation = ApprovalRecommendation(
                    recommendation_type="OUT_OF_SCOPE",
                    summary=summary,
                    risk_level="LOW",
                    reasons=[intent.reason],
                    required_actions=["请输入一条采购需求，例如：行政部采购 20 把办公椅，预算 8000 元，用于新员工工位补充。"],
                    human_review_required=False,
                )
                run.completed_at = utc_now()
                self._trace(
                    run,
                    "RUN_COMPLETED",
                    "非采购请求已边界处理",
                    "闲聊或无关请求不会进入 RAG 和业务工具调用流程。",
                    {"intent": intent.model_dump()},
                    run.recommendation.model_dump(),
                )
                return self.store.save(run)

            self._set_state(run, "EXTRACTING_REQUIREMENTS")
            extraction = self.extractor.extract(run.original_input)
            purchase = extraction.purchase
            if intent and intent.intent == "compliance_risk" and "USER_COMPLIANCE_RISK" not in purchase.flags:
                purchase.flags.append("USER_COMPLIANCE_RISK")
            run.purchase_request = purchase
            for record in extraction.llm_records:
                self._trace_llm(run, record)
            self._trace(
                run,
                "FIELDS_EXTRACTED",
                "完成采购字段抽取",
                "从自然语言中抽取部门、品类、金额、供应商、用途等字段。",
                {"raw_input": run.original_input},
                {
                    "extractor_source": extraction.source,
                    "fallback_used": extraction.fallback_used,
                    "purchase": purchase.model_dump(),
                },
            )

            questions = self._build_missing_questions(purchase)
            run.missing_questions = questions
            blocking_questions = [question for question in questions if question.required]
            if blocking_questions:
                self._set_state(run, "NEED_INFO")
                run.recommendation = ApprovalRecommendation(
                    recommendation_type="NEED_MORE_INFO",
                    summary="当前信息不足，需补充关键字段后继续预审。",
                    risk_level="MEDIUM",
                    reasons=[question.reason for question in blocking_questions],
                    required_actions=[question.question_text for question in blocking_questions],
                    human_review_required=False,
                )
                self._trace(
                    run,
                    "MISSING_INFO_DETECTED",
                    "发现阻塞预审的缺失信息",
                    "信息不足时不继续生成确定性审批建议。",
                    purchase.model_dump(),
                    {"questions": [question.model_dump() for question in blocking_questions]},
                )
                return self.store.save(run)
            if questions:
                self._trace(
                    run,
                    "MISSING_INFO_DETECTED",
                    "记录非阻塞补充信息",
                    "部分材料或确认项不阻塞预审，但会进入最终建议的下一步动作。",
                    purchase.model_dump(),
                    {"questions": [question.model_dump() for question in questions]},
                )

            self._set_state(run, "RETRIEVING_POLICY")
            run.policy_citations = self.retriever.retrieve(purchase, run.original_input)
            self._trace(
                run,
                "POLICY_RETRIEVED",
                "检索采购制度依据",
                "根据采购品类、金额、供应商和紧急程度检索制度片段。",
                {"query_context": self._query_context(purchase)},
                {"citations": [citation.model_dump() for citation in run.policy_citations]},
            )

            self._set_state(run, "PLANNING_TOOLS")
            tool_plan = self._plan_tools(purchase)
            self._trace(
                run,
                "TOOL_PLANNED",
                "生成工具调用计划",
                "根据抽取字段决定预算、供应商、历史价格和审批链工具调用。",
                purchase.model_dump(),
                {"tool_plan": tool_plan},
            )

            self._set_state(run, "CALLING_TOOLS")
            run.tool_calls = [self._call_tool(name, reason, purchase) for name, reason in tool_plan]
            for call in run.tool_calls:
                self._trace(
                    run,
                    "TOOL_CALLED",
                    f"调用{call.tool_name}",
                    call.call_reason,
                    call.input_args,
                    call.output_data,
                )

            self._set_state(run, "ASSESSING_RISK")
            run.risk_findings = self._assess_risks(run)
            approval_call = self._maybe_call_approval_chain(run)
            if approval_call:
                run.tool_calls.append(approval_call)
                approval_risk = self._approval_risk_from_call(approval_call, purchase)
                if approval_risk:
                    run.risk_findings.append(approval_risk)
            self._trace(
                run,
                "RISK_ASSESSED",
                "完成采购风险评估",
                "综合制度引用、工具结果和用户输入识别风险项。",
                {
                    "citations": [citation.citation_id for citation in run.policy_citations],
                    "tool_calls": [call.tool_call_id for call in run.tool_calls],
                },
                {"risk_findings": [risk.model_dump() for risk in run.risk_findings]},
            )

            self._set_state(run, "GENERATING_RECOMMENDATION")
            run.recommendation = self._generate_recommendation(run)
            self._polish_recommendation(run)
            self._trace(
                run,
                "RECOMMENDATION_GENERATED",
                "生成结构化审批建议",
                "审批建议区分制度依据、工具结果、风险项和下一步动作。",
                {"risk_findings": [risk.risk_id for risk in run.risk_findings]},
                run.recommendation.model_dump(),
            )

            if run.recommendation.human_review_required:
                self._set_state(run, "WAITING_HUMAN_APPROVAL")
                role = run.recommendation.target_reviewer_roles[0] if run.recommendation.target_reviewer_roles else "采购专员"
                run.human_review = HumanReview(
                    reason=run.recommendation.summary,
                    reviewer_role=role,
                )
                self._trace(
                    run,
                    "HUMAN_REVIEW_REQUESTED",
                    "进入人工确认节点",
                    "高风险或依据不足场景暂停自动流程，等待人工确认。",
                    run.recommendation.model_dump(),
                    run.human_review.model_dump(),
                )
            else:
                self._set_state(run, "COMPLETED")
                run.completed_at = utc_now()
                self._trace(
                    run,
                    "RUN_COMPLETED",
                    "采购预审完成",
                    "未触发强制人工确认，输出预审建议。",
                    {},
                    {"recommendation_type": run.recommendation.recommendation_type},
                )

            return self.store.save(run)
        except Exception as exc:
            self._set_state(run, "FAILED")
            self._trace(
                run,
                "ERROR_OCCURRED",
                "运行失败",
                "Agent 工作流执行过程中出现异常。",
                {},
                {"error": str(exc)},
            )
            return self.store.save(run)

    def apply_human_review(self, run_id: str, request: HumanReviewRequest) -> AgentRun:
        run = self.store.get(run_id)
        if run is None:
            raise KeyError(f"Run {run_id} not found")

        if run.human_review is None:
            run.human_review = HumanReview(
                required=True,
                reason="人工主动补充确认结果。",
                reviewer_role=request.reviewer_role,
            )

        status_map = {
            "APPROVE": "APPROVED",
            "REJECT": "REJECTED",
            "REQUEST_MORE_INFO": "MORE_INFO_REQUIRED",
            "TRANSFER": "TRANSFERRED",
        }
        run.human_review.status = status_map[request.decision]  # type: ignore[assignment]
        run.human_review.decision = request.decision
        run.human_review.comment = request.comment
        run.human_review.reviewer_role = request.reviewer_role
        run.human_review.reviewed_at = utc_now()

        self._trace(
            run,
            "HUMAN_DECISION_RECEIVED",
            "收到人工确认结果",
            "人工确认结果写入 trace，并驱动流程恢复或结束。",
            request.model_dump(),
            run.human_review.model_dump(),
        )

        if request.decision == "REQUEST_MORE_INFO":
            self._set_state(run, "NEED_INFO")
            run.missing_questions = [
                MissingInfoQuestion(
                    field_name="human_requested_materials",
                    question_text=request.comment or "请补充人工复核要求的材料。",
                    required=True,
                    reason="人工复核要求补充材料后再继续预审。",
                )
            ]
        elif request.decision == "TRANSFER":
            self._set_state(run, "WAITING_HUMAN_APPROVAL")
        else:
            self._set_state(run, "COMPLETED")
            run.completed_at = utc_now()
            if run.recommendation and request.decision == "REJECT":
                run.recommendation.recommendation_type = "REJECT_RECOMMENDED"  # type: ignore[assignment]
                run.recommendation.summary = "人工复核后不建议继续提交本次采购。"
                run.recommendation.human_review_required = False
            elif run.recommendation and request.decision == "APPROVE":
                run.recommendation.recommendation_type = "HUMAN_APPROVED_TO_CONTINUE"  # type: ignore[assignment]
                run.recommendation.summary = "人工复核已允许继续，后续仍需按建议补充材料并进入正式审批。"
                run.recommendation.human_review_required = False

        return self.store.save(run)

    def _reset_for_resume(self, run: AgentRun) -> None:
        run.missing_questions = []
        run.policy_citations = []
        run.tool_calls = []
        run.risk_findings = []
        run.recommendation = None
        run.human_review = None
        run.completed_at = None

    def _set_state(self, run: AgentRun, state: AgentState) -> None:
        run.current_state = state
        run.updated_at = utc_now()

    def _trace(
        self,
        run: AgentRun,
        event_type: str,
        title: str,
        detail: str,
        input_snapshot: dict[str, Any],
        output_snapshot: dict[str, Any],
    ) -> None:
        run.trace_events.append(
            AgentTraceEvent(
                run_id=run.run_id,
                state=run.current_state,
                event_type=event_type,
                title=title,
                detail=detail,
                input_snapshot=input_snapshot,
                output_snapshot=output_snapshot,
            )
        )

    def _classify_intent(self, run: AgentRun, message: str) -> LLMIntentResult | None:
        intent, record = self.llm_provider.classify_intent(message)
        self._trace_llm(run, record)
        if intent:
            return intent

        if any(word in message for word in ["拆单", "拆成", "拆分", "不用走", "绕过审批", "规避审批"]):
            return LLMIntentResult(intent="compliance_risk", reason="规则兜底识别到规避审批或拆单意图。")
        if not any(word in message for word in ["采购", "买", "预算", "金额", "供应商", "设备", "办公", "审批", "显示器", "MacBook"]):
            if any(word in message for word in ["你好", "您好", "你能做什么", "介绍"]):
                return LLMIntentResult(intent="chitchat", reason="规则兜底识别为能力咨询或闲聊。")
            return LLMIntentResult(intent="out_of_scope", reason="规则兜底识别为非采购预审请求。")
        return LLMIntentResult(intent="procurement", reason="规则兜底识别为采购预审相关输入。")

    def _boundary_summary(self, run: AgentRun, message: str, intent: LLMIntentResult) -> str:
        reply, record = self.llm_provider.generate_boundary_reply(message, intent.intent, intent.reason)
        rejection_reason = validate_boundary_reply(reply)
        if rejection_reason:
            record.fallback_used = True
            record.error_message = f"boundary reply rejected: {rejection_reason}"
            record.output_summary["accepted"] = False
            record.output_summary["rejected_reason"] = rejection_reason
            self._trace_llm(run, record)
            self._trace(
                run,
                "LLM_BOUNDARY_REPLY_REJECTED",
                "丢弃 LLM 边界回复",
                "LLM 返回的非采购边界回复越过 ProcurePilot 能力范围，保留规则兜底文案。",
                {"intent": intent.model_dump()},
                {"rejected_reason": rejection_reason},
            )
            return BOUNDARY_FALLBACK_SUMMARY

        if reply:
            record.output_summary["accepted"] = True
            self._trace_llm(run, record)
            return reply

        self._trace_llm(run, record)
        return BOUNDARY_FALLBACK_SUMMARY

    def _trace_llm(self, run: AgentRun, record: LLMCallRecord) -> None:
        self._trace(
            run,
            "LLM_CALLED",
            f"LLM 调用：{record.purpose}",
            "OpenAI-compatible LLMProvider 调用记录；API Key 不会写入 trace。",
            {"purpose": record.purpose, "enabled": record.enabled},
            {
                "model": record.model,
                "success": record.success,
                "elapsed_ms": record.elapsed_ms,
                "fallback_used": record.fallback_used,
                "token_usage": record.token_usage,
                "error_message": record.error_message,
                "output_summary": record.output_summary,
            },
        )

    def _build_missing_questions(self, purchase: PurchaseRequest) -> list[MissingInfoQuestion]:
        questions: list[MissingInfoQuestion] = []
        if "USER_COMPLIANCE_RISK" in purchase.flags:
            return questions

        if not purchase.department:
            questions.append(
                MissingInfoQuestion(
                    field_name="department",
                    question_text="请补充申请部门或业务归属。",
                    reason="缺少申请部门，无法查询预算余额和审批归属。",
                )
            )
        if not purchase.purchase_category:
            questions.append(
                MissingInfoQuestion(
                    field_name="purchase_category",
                    question_text="请补充采购品类或具体物品名称。",
                    reason="缺少采购品类，无法检索适用制度和判断是否需要专项复核。",
                )
            )
        if purchase.purchase_category in {"IT设备", "办公用品", "直播设备"} and purchase.quantity is None:
            blocking = purchase.amount is None or not purchase.department
            questions.append(
                MissingInfoQuestion(
                    field_name="quantity",
                    question_text="请补充本次采购数量。",
                    required=blocking,
                    reason="缺少数量，无法判断预算单价、历史价格偏离和采购规模是否合理。",
                )
            )
        if purchase.amount is None:
            questions.append(
                MissingInfoQuestion(
                    field_name="amount",
                    question_text="请补充本次采购预算金额。",
                    reason="缺少预算金额，无法判断金额阈值、预算风险和审批链。",
                )
            )
        if not purchase.purpose and not purchase.vendor_name:
            questions.append(
                MissingInfoQuestion(
                    field_name="purpose",
                    question_text="请补充本次采购用途或业务背景。",
                    reason="缺少采购用途，无法确认预算科目和业务必要性。",
                )
            )
        if not purchase.budget_category:
            questions.append(
                MissingInfoQuestion(
                    field_name="budget_category",
                    question_text="请补充或确认本次采购对应的预算科目。",
                    required=True,
                    reason="缺少预算科目，无法进行预算余额和科目匹配校验。",
                )
            )
        elif not purchase.budget_category_confirmed:
            questions.append(
                MissingInfoQuestion(
                    field_name="budget_category_confirmed",
                    question_text=f"请确认本次采购是否使用“{purchase.budget_category}”预算科目。",
                    required=False,
                    reason="预算科目由 Agent 推断，需用户确认后才能作为正式申请材料。",
                )
            )
        if purchase.amount is not None and purchase.amount >= 100000:
            questions.append(
                MissingInfoQuestion(
                    field_name="comparison_materials",
                    question_text="请确认是否已有三方比价材料或价格合理性说明。",
                    required=False,
                    reason="金额达到采购复核范围，提交前通常需要补充比价或价格说明材料。",
                )
            )
            questions.append(
                MissingInfoQuestion(
                    field_name="quote_materials",
                    question_text="请确认是否已有供应商报价或报价单。",
                    required=False,
                    reason="高金额采购需要保留报价依据，便于采购专员复核。",
                )
            )
            if purchase.vendor_name:
                questions.append(
                    MissingInfoQuestion(
                        field_name="specified_vendor_reason",
                        question_text="请补充指定供应商原因。",
                        required=False,
                        reason="已指定供应商，需说明指定原因并保留供应商资质材料。",
                    )
                )
            else:
                questions.append(
                    MissingInfoQuestion(
                        field_name="vendor_selection",
                        question_text="请确认是否已指定供应商，或是否接受采购部推荐供应商。",
                        required=False,
                        reason="高金额采购需要明确供应商选择方式，便于后续审批。",
                    )
                )
        if "POSSIBLE_AMOUNT_CONFLICT" in purchase.flags:
            questions.append(
                MissingInfoQuestion(
                    field_name="amount",
                    question_text="请确认预算金额和总价是否一致。",
                    reason="输入中可能同时出现预算和总价，需确认金额口径。",
                )
            )
        return questions[:8]

    def _query_context(self, purchase: PurchaseRequest) -> dict[str, Any]:
        return {
            "department": purchase.department,
            "category": purchase.purchase_category,
            "item": purchase.item_name,
            "amount": purchase.amount,
            "vendor": purchase.vendor_name,
            "urgent": purchase.is_urgent,
            "flags": purchase.flags,
        }

    def _plan_tools(self, purchase: PurchaseRequest) -> list[tuple[str, str]]:
        plan: list[tuple[str, str]] = []
        if purchase.department and purchase.amount is not None and purchase.budget_category:
            plan.append(("budget_lookup", "已识别部门、金额和预算科目，需要确认预算余额与科目匹配。"))
        if purchase.vendor_name:
            plan.append(("vendor_qualification_lookup", "用户指定供应商，需要确认供应商准入和资质状态。"))
            plan.append(("vendor_risk_lookup", "用户指定供应商，需要查询黑名单、观察名单和履约异常。"))
        if purchase.item_name and purchase.amount is not None and purchase.quantity and (
            purchase.amount >= 50000 or purchase.purchase_category == "IT设备"
        ):
            plan.append(("historical_price_lookup", "金额或品类触发价格合理性校验，需要查询历史采购价格。"))
        return plan

    def _call_tool(
        self,
        tool_name: str,
        reason: str,
        purchase: PurchaseRequest,
        risk_types: list[str] | None = None,
    ) -> ToolCallRecord:
        input_args = purchase.model_dump()
        try:
            if tool_name == "budget_lookup":
                output = self.tools.query_budget(purchase)
            elif tool_name == "vendor_qualification_lookup":
                output = self.tools.query_vendor_qualification(purchase)
            elif tool_name == "vendor_risk_lookup":
                output = self.tools.query_vendor_risk(purchase)
            elif tool_name == "historical_price_lookup":
                output = self.tools.query_historical_price(purchase)
            elif tool_name == "approval_chain_lookup":
                output = self.tools.query_approval_chain(purchase, risk_types or self._preliminary_risk_types(purchase))
            else:
                output = {"ok": False, "message": f"未知工具：{tool_name}", "risk_level": "HIGH"}
        except Exception as exc:
            return ToolCallRecord(
                tool_name=tool_name,
                call_reason=reason,
                input_args=input_args,
                output_summary="工具调用失败。",
                output_data={},
                status="FAILED",
                error_message=str(exc),
                risk_impact="关键业务工具失败，需人工复核。",
            )

        status = "SUCCESS" if output.get("ok", True) else "FAILED"
        return ToolCallRecord(
            tool_name=tool_name,
            call_reason=reason,
            input_args=input_args,
            output_summary=output.get("message", "工具调用完成。"),
            output_data=output,
            status=status,  # type: ignore[arg-type]
            error_message=None if status == "SUCCESS" else output.get("message"),
            risk_impact=output.get("risk_level") or output.get("approval_level"),
        )

    def _preliminary_risk_types(self, purchase: PurchaseRequest) -> list[str]:
        risk_types: list[str] = []
        if purchase.amount is not None and purchase.amount >= 100000:
            risk_types.append("AMOUNT_RISK")
        if purchase.vendor_name:
            risk_types.append("VENDOR_QUALIFICATION_RISK")
        if "USER_COMPLIANCE_RISK" in purchase.flags:
            risk_types.append("USER_COMPLIANCE_RISK")
        return risk_types

    def _maybe_call_approval_chain(self, run: AgentRun) -> ToolCallRecord | None:
        purchase = run.purchase_request
        if purchase is None:
            return None

        risk_types = self._risk_types_for_approval(run.risk_findings)
        needs_approval = (
            bool(risk_types)
            or bool(purchase.amount and purchase.amount >= 100000)
            or bool(purchase.vendor_name and any(risk.requires_human_review for risk in run.risk_findings))
            or "USER_COMPLIANCE_RISK" in purchase.flags
        )
        if not needs_approval:
            return None

        self._set_state(run, "PLANNING_TOOLS")
        self._trace(
            run,
            "TOOL_PLANNED",
            "补充审批链工具调用计划",
            "基于已完成的预算、供应商、历史价格和制度风险结果查询审批链。",
            {"risk_types": risk_types},
            {"tool": "approval_chain_lookup"},
        )
        self._set_state(run, "CALLING_TOOLS")
        call = self._call_tool(
            "approval_chain_lookup",
            "基于真实风险评估结果查询需要转交的审批角色。",
            purchase,
            risk_types=risk_types,
        )
        self._trace(
            run,
            "TOOL_CALLED",
            f"调用{call.tool_name}",
            call.call_reason,
            call.input_args,
            call.output_data,
        )
        self._set_state(run, "ASSESSING_RISK")
        return call

    def _risk_types_for_approval(self, risks: list[RiskFinding]) -> list[str]:
        material = [
            risk.risk_type
            for risk in risks
            if risk.requires_human_review or RISK_ORDER[risk.risk_level] >= RISK_ORDER["HIGH"]
        ]
        return list(dict.fromkeys(material))

    def _approval_risk_from_call(self, call: ToolCallRecord, purchase: PurchaseRequest) -> RiskFinding | None:
        if call.status != "SUCCESS":
            return RiskFinding(
                risk_type="TOOL_FAILURE_RISK",
                risk_level="HIGH",
                title="审批链工具失败",
                description="审批链工具未能返回可靠结果，不能由 Agent 猜测审批角色。",
                evidence_type="工具",
                evidence_refs=[call.tool_call_id],
                requires_human_review=True,
                suggested_action="转采购专员人工确认审批路径。",
            )
        if not call.output_data.get("approval_required"):
            return None
        return RiskFinding(
            risk_type="APPROVAL_PATH_RISK",
            risk_level="MEDIUM",
            title="需按审批链转交",
            description=call.output_data.get("message", "审批链工具返回需人工确认角色。"),
            evidence_type="工具",
            evidence_refs=[call.tool_call_id],
            requires_human_review=bool(purchase.amount and purchase.amount >= 100000)
            or any(role in call.output_data.get("approvers", []) for role in ["采购专员", "合规人员", "预算负责人"]),
            suggested_action="按工具返回的审批角色转交。",
        )

    def _assess_risks(self, run: AgentRun) -> list[RiskFinding]:
        purchase = run.purchase_request
        if purchase is None:
            return []

        risks: list[RiskFinding] = []

        if not any(citation.supports_conclusion for citation in run.policy_citations):
            self._add_risk(
                risks,
                "POLICY_EVIDENCE_INSUFFICIENT",
                "HIGH",
                "制度依据不足",
                "未检索到可支撑本次采购合规判断的制度引用，不能输出确定性合规结论。",
                "RAG",
                [],
                True,
                "转采购专员人工复核制度适用性。",
            )

        if "USER_COMPLIANCE_RISK" in purchase.flags:
            refs = self._citation_refs(run, "USER_COMPLIANCE_RISK")
            self._add_policy_risk(
                run,
                risks,
                risk_type="USER_COMPLIANCE_RISK",
                risk_level="BLOCKED",
                title="疑似规避审批请求",
                description="用户输入包含拆单、拆分或绕过审批意图，系统不能协助规避采购制度。",
                refs=refs,
                requires_human_review=True,
                suggested_action="停止拆单方案，按真实采购金额走正常审批并转合规人员确认。",
            )

        if purchase.amount is not None:
            if purchase.amount > 500000:
                self._add_policy_risk(
                    run,
                    risks,
                    risk_type="APPROVAL_PATH_RISK",
                    risk_level="HIGH",
                    title="金额触发高额采购审批",
                    description=f"本次预算金额 {purchase.amount:.0f} 元，超过 50 万元，需进入高额采购审批流程。",
                    refs=self._citation_refs(run, "APPROVAL_PATH_RISK"),
                    requires_human_review=True,
                    suggested_action="提交采购负责人、预算负责人及相关业务负责人共同确认。",
                )
            elif purchase.amount >= 100000:
                self._add_policy_risk(
                    run,
                    risks,
                    risk_type="AMOUNT_RISK",
                    risk_level="HIGH",
                    title="金额触发采购复核",
                    description=f"本次预算金额 {purchase.amount:.0f} 元，达到 10 万元及以上采购复核范围。",
                    refs=self._citation_refs(run, "AMOUNT_RISK"),
                    requires_human_review=True,
                    suggested_action="补充比价材料或价格合理性说明，并提交采购专员复核。",
                )
            elif purchase.amount >= 10000:
                self._add_policy_risk(
                    run,
                    risks,
                    risk_type="AMOUNT_RISK",
                    risk_level="MEDIUM",
                    title="金额需常规审批",
                    description=f"本次预算金额 {purchase.amount:.0f} 元，高于低额采购范围。",
                    refs=self._citation_refs(run, "AMOUNT_RISK"),
                    requires_human_review=False,
                    suggested_action="按常规采购流程提交部门负责人确认。",
                )

        if purchase.purchase_category == "IT设备" and purchase.amount is not None and purchase.amount >= 50000:
            self._add_policy_risk(
                run,
                risks,
                risk_type="CATEGORY_POLICY_RISK",
                risk_level="HIGH" if purchase.amount >= 100000 else "MEDIUM",
                title="IT 设备采购需专项确认",
                description="本次采购属于 IT 设备，制度要求由 IT 部门确认型号、配置和业务必要性。",
                refs=self._citation_refs(run, "CATEGORY_POLICY_RISK"),
                requires_human_review=purchase.amount >= 100000,
                suggested_action="补充设备型号、配置清单和 IT 部门确认意见。",
            )

        if purchase.vendor_name:
            self._add_policy_risk(
                run,
                risks,
                risk_type="VENDOR_QUALIFICATION_RISK",
                risk_level="MEDIUM",
                title="指定供应商需说明原因",
                description=f"用户指定 {purchase.vendor_name} 供货，需保留指定供应商原因和资质材料。",
                refs=self._citation_refs(run, "VENDOR_QUALIFICATION_RISK"),
                requires_human_review=bool(purchase.amount and purchase.amount >= 100000),
                suggested_action="补充指定供应商原因和准入资质材料。",
            )

        if not purchase.purpose and "USER_COMPLIANCE_RISK" not in purchase.flags:
            self._add_risk(
                risks,
                "MISSING_INFO",
                "MEDIUM",
                "采购用途仍需补充",
                "本次输入未明确采购用途，可能影响预算科目和业务必要性判断。",
                "用户输入",
                [],
                False,
                "提交前补充采购用途或业务背景。",
            )

        for call in run.tool_calls:
            data = call.output_data
            if call.status == "FAILED" and call.tool_name in {"budget_lookup", "vendor_risk_lookup", "approval_chain_lookup"}:
                self._add_risk(
                    risks,
                    "TOOL_FAILURE_RISK",
                    "HIGH",
                    "关键业务工具失败",
                    f"{call.tool_name} 未能返回可靠结果，不能由 Agent 猜测业务状态。",
                    "工具",
                    [call.tool_call_id],
                    True,
                    "转人工复核对应业务状态。",
                )

            if call.tool_name == "budget_lookup" and call.status == "SUCCESS":
                if not data.get("is_sufficient") or not data.get("category_matched"):
                    self._add_risk(
                        risks,
                        "BUDGET_RISK",
                        "HIGH",
                        "预算不足或科目不匹配",
                        data.get("message", "预算工具显示存在风险。"),
                        "工具",
                        [call.tool_call_id],
                        True,
                        "由预算负责人确认预算调整或补充预算。",
                    )

            if call.tool_name == "vendor_qualification_lookup" and call.status == "SUCCESS":
                if not data.get("is_approved") or data.get("risk_level") == "HIGH":
                    self._add_risk(
                        risks,
                        "VENDOR_QUALIFICATION_RISK",
                        "HIGH",
                        "供应商准入或资质风险",
                        data.get("message", "供应商资质工具显示存在风险。"),
                        "工具",
                        [call.tool_call_id],
                        True,
                        "更换准入供应商或补充资质材料后由采购专员复核。",
                    )

            if call.tool_name == "vendor_risk_lookup" and call.status == "SUCCESS":
                level = data.get("risk_level")
                if data.get("blacklist_hit") or level == "HIGH":
                    self._add_risk(
                        risks,
                        "VENDOR_BLACKLIST_RISK",
                        "HIGH",
                        "供应商存在异常风险",
                        data.get("message", "供应商风险工具命中异常记录。"),
                        "工具",
                        [call.tool_call_id],
                        True,
                        "暂缓提交，转采购专员和合规人员复核供应商风险。",
                    )
                elif level == "MEDIUM":
                    self._add_risk(
                        risks,
                        "VENDOR_BLACKLIST_RISK",
                        "MEDIUM",
                        "供应商存在观察风险",
                        data.get("message", "供应商风险工具提示观察风险。"),
                        "工具",
                        [call.tool_call_id],
                        True,
                        "由采购专员确认是否需要替换供应商。",
                    )

            if call.tool_name == "historical_price_lookup" and call.status == "SUCCESS":
                level = data.get("risk_level")
                if level in {"HIGH", "MEDIUM"}:
                    self._add_risk(
                        risks,
                        "PRICE_ANOMALY_RISK",
                        level,
                        "价格高于历史均价",
                        data.get("message", "历史价格工具显示价格偏离。"),
                        "工具",
                        [call.tool_call_id],
                        level == "HIGH" or bool(purchase.amount and purchase.amount >= 100000),
                        "补充价格合理性说明和比价材料。",
                    )

            if call.tool_name == "approval_chain_lookup" and call.status == "SUCCESS" and data.get("approval_required"):
                self._add_risk(
                    risks,
                    "APPROVAL_PATH_RISK",
                    "MEDIUM",
                    "需按审批链转交",
                    data.get("message", "审批链工具返回需人工确认角色。"),
                    "工具",
                    [call.tool_call_id],
                    bool(purchase.amount and purchase.amount >= 100000),
                    "按工具返回的审批角色转交。",
                )

        return risks

    def _add_risk(
        self,
        risks: list[RiskFinding],
        risk_type: str,
        risk_level: str,
        title: str,
        description: str,
        evidence_type: str,
        evidence_refs: list[str],
        requires_human_review: bool,
        suggested_action: str,
    ) -> None:
        risks.append(
            RiskFinding(
                risk_type=risk_type,
                risk_level=risk_level,  # type: ignore[arg-type]
                title=title,
                description=description,
                evidence_type=evidence_type,
                evidence_refs=evidence_refs,
                requires_human_review=requires_human_review,
                suggested_action=suggested_action,
            )
        )

    def _add_policy_risk(
        self,
        run: AgentRun,
        risks: list[RiskFinding],
        risk_type: str,
        risk_level: str,
        title: str,
        description: str,
        refs: list[str],
        requires_human_review: bool,
        suggested_action: str,
    ) -> None:
        if not refs:
            self._add_risk(
                risks,
                "POLICY_EVIDENCE_INSUFFICIENT",
                "HIGH",
                "制度引用不足",
                f"未找到可支撑“{title}”判断的制度引用，不能输出确定性制度结论。",
                "RAG",
                [],
                True,
                "转采购专员人工复核制度适用性。",
            )
            return
        self._add_risk(
            risks,
            risk_type,
            risk_level,
            title,
            description,
            "制度",
            refs,
            requires_human_review,
            suggested_action,
        )

    def _citation_refs(self, run: AgentRun, risk_type: str) -> list[str]:
        return [
            citation.citation_id
            for citation in run.policy_citations
            if citation.risk_type == risk_type and citation.supports_conclusion
        ]

    def _generate_recommendation(self, run: AgentRun) -> ApprovalRecommendation:
        max_level = self._max_risk_level(run.risk_findings)
        human_required = any(risk.requires_human_review for risk in run.risk_findings)
        blocked = max_level == "BLOCKED"
        roles = self._reviewer_roles(run)

        high_or_medium = [risk for risk in run.risk_findings if RISK_ORDER[risk.risk_level] >= RISK_ORDER["MEDIUM"]]
        reasons = [risk.description for risk in high_or_medium] or [
            "制度引用和业务工具结果未发现阻塞性风险。",
            "预算校验和金额阈值判断支持继续提交。",
        ]
        actions = list(dict.fromkeys([risk.suggested_action for risk in high_or_medium]))
        advisory_actions = [
            question.question_text
            for question in run.missing_questions
            if not question.required and question.status == "PENDING"
        ]
        actions = list(dict.fromkeys(actions + advisory_actions))

        if blocked:
            recommendation_type = "REJECT_RECOMMENDED"
            summary = "识别到疑似规避采购制度的请求，建议停止当前提交并转人工合规复核。"
            human_required = True
            if "合规人员" not in roles:
                roles.append("合规人员")
        elif human_required:
            recommendation_type = "HUMAN_REVIEW_REQUIRED"
            summary = "本次采购存在需人工确认的风险，Agent 仅给出预审建议，不能直接放行。"
        elif max_level == "MEDIUM":
            recommendation_type = "SUBMIT_AFTER_SUPPLEMENT"
            summary = "本次采购可在补充说明或材料后提交常规审批。"
        else:
            recommendation_type = "SUBMIT_RECOMMENDED"
            summary = "本次采购预审未发现高风险，建议按低额或常规流程提交。"
            actions = actions or ["按系统生成的采购摘要和制度引用提交申请。"]

        return ApprovalRecommendation(
            recommendation_type=recommendation_type,  # type: ignore[arg-type]
            summary=summary,
            risk_level=max_level,
            reasons=reasons,
            required_actions=actions,
            policy_citation_ids=[citation.citation_id for citation in run.policy_citations],
            tool_call_ids=[call.tool_call_id for call in run.tool_calls],
            human_review_required=human_required,
            target_reviewer_roles=roles if human_required else [],
        )

    def _polish_recommendation(self, run: AgentRun) -> None:
        if run.recommendation is None:
            return

        context = {
            "original_summary": run.recommendation.summary,
            "recommendation_type": run.recommendation.recommendation_type,
            "risk_level": run.recommendation.risk_level,
            "human_review_required": run.recommendation.human_review_required,
            "purchase_request": run.purchase_request.model_dump() if run.purchase_request else None,
            "risk_findings": [risk.model_dump() for risk in run.risk_findings],
            "policy_citations": [
                citation.model_dump()
                for citation in run.policy_citations
                if citation.supports_conclusion
            ],
            "tool_results": [
                {
                    "tool_name": call.tool_name,
                    "status": call.status,
                    "output_summary": call.output_summary,
                }
                for call in run.tool_calls
            ],
        }
        summary, record = self.llm_provider.polish_recommendation(context)
        rejection_reason = validate_polished_summary(run.recommendation, summary)
        if rejection_reason:
            record.fallback_used = True
            record.error_message = f"summary rejected: {rejection_reason}"
            record.output_summary["accepted"] = False
            record.output_summary["rejected_reason"] = rejection_reason
            self._trace_llm(run, record)
            self._trace(
                run,
                "LLM_SUMMARY_REJECTED",
                "丢弃 LLM 润色摘要",
                "LLM 返回的 summary 与结构化审批结论存在冲突，保留规则生成的 summary。",
                {
                    "recommendation_type": run.recommendation.recommendation_type,
                    "risk_level": run.recommendation.risk_level,
                    "human_review_required": run.recommendation.human_review_required,
                },
                {"rejected_reason": rejection_reason},
            )
            return

        if summary:
            record.output_summary["accepted"] = True
            self._trace_llm(run, record)
            run.recommendation.summary = summary
            return
        self._trace_llm(run, record)

    def _max_risk_level(self, risks: list[RiskFinding]) -> str:
        if not risks:
            return "LOW"
        return max((risk.risk_level for risk in risks), key=lambda level: RISK_ORDER[level])

    def _reviewer_roles(self, run: AgentRun) -> list[str]:
        roles: list[str] = []
        approval_call = next((call for call in run.tool_calls if call.tool_name == "approval_chain_lookup"), None)
        if approval_call:
            roles.extend(approval_call.output_data.get("approvers", []))

        for risk in run.risk_findings:
            if risk.risk_type == "BUDGET_RISK" and risk.requires_human_review:
                roles.append("预算负责人")
            if risk.risk_type in {"VENDOR_BLACKLIST_RISK", "VENDOR_QUALIFICATION_RISK"} and risk.requires_human_review:
                roles.extend(["采购专员", "合规人员"])
            if risk.risk_type == "CATEGORY_POLICY_RISK" and risk.requires_human_review:
                roles.append("IT 负责人")
            if risk.risk_type == "USER_COMPLIANCE_RISK":
                roles.append("合规人员")

        return list(dict.fromkeys(roles))


def validate_polished_summary(recommendation: ApprovalRecommendation, summary: str | None) -> str | None:
    if not summary:
        return None

    normalized = "".join(summary.split())
    direct_approval_phrases = [
        "可直接提交",
        "可以直接提交",
        "建议直接提交",
        "直接提交即可",
        "直接放行",
        "自动通过",
        "可自动通过",
    ]
    no_review_phrases = [
        "无需复核",
        "不需要复核",
        "无需人工",
        "不需要人工",
        "无需审批",
        "不需要审批",
    ]
    low_risk_phrases = ["低风险", "无风险", "未发现风险", "未发现高风险", "风险较低", "风险可忽略"]
    continue_phrases = ["建议继续提交", "可继续提交", "可以继续提交", "允许继续提交", "同意继续提交"]
    complete_info_phrases = ["材料完整", "资料完整", "信息完整", "无需补充", "不需要补充", "材料齐全"]

    if recommendation.human_review_required or recommendation.recommendation_type == "HUMAN_REVIEW_REQUIRED":
        if contains_unnegated_any(normalized, direct_approval_phrases) or contains_any(normalized, no_review_phrases):
            return "需要人工确认的结论不能表达为可直接提交、无需复核或自动通过"
        if not contains_any(normalized, ["人工", "复核", "确认", "审批", "不能直接", "不得直接", "暂缓", "暂停", "转"]):
            return "需要人工确认的结论必须保留人工复核或审批提示"

    if recommendation.risk_level in {"HIGH", "BLOCKED"} and contains_any(normalized, low_risk_phrases):
        return "高风险或阻断结论不能表达为低风险或未发现高风险"

    if recommendation.recommendation_type == "REJECT_RECOMMENDED":
        if (
            contains_unnegated_any(normalized, direct_approval_phrases)
            or contains_any(normalized, no_review_phrases + continue_phrases)
        ):
            return "拒绝或阻断结论不能表达为建议继续提交"
        if not contains_any(normalized, ["拒绝", "停止", "不建议", "不能", "不得", "禁止", "暂停", "暂缓", "合规"]):
            return "拒绝或阻断结论必须保留停止、拒绝或合规复核提示"

    if recommendation.recommendation_type == "NEED_MORE_INFO":
        if (
            contains_unnegated_any(normalized, direct_approval_phrases)
            or contains_any(normalized, no_review_phrases + complete_info_phrases)
        ):
            return "信息不足结论不能表达为材料完整或无需补充"
        if not contains_any(normalized, ["补充", "信息不足", "缺少", "需确认", "请确认"]):
            return "信息不足结论必须保留补充或确认提示"

    return None


def validate_boundary_reply(reply: str | None) -> str | None:
    if not reply:
        return None

    normalized = "".join(reply.split())
    unsupported_claims = [
        "已完成采购预审",
        "已经完成采购预审",
        "完成了采购预审",
        "已完成预审",
        "已经完成预审",
        "已查询预算",
        "已经查询预算",
        "预算充足",
        "预算不足",
        "已查询供应商",
        "已经查询供应商",
        "供应商合格",
        "供应商不合格",
        "供应商已通过",
        "已查询审批链",
        "已经查询审批链",
        "已调用工具",
        "已经调用工具",
        "已完成审批",
        "审批已完成",
        "已通过审批",
        "审批通过",
        "制度引用如下",
        "根据制度第",
        "历史价格正常",
        "价格合理",
        "风险等级为",
        "建议提交本次采购",
    ]
    unrelated_fulfillment = [
        "以下是一首诗",
        "给你写一首诗",
        "我为你写一首诗",
        "诗如下",
        "这首诗",
        "下面是一首诗",
    ]
    unsafe_compliance_phrases = [
        "帮你规避审批",
        "可以规避审批",
        "能够规避审批",
        "绕过采购制度",
        "不用走审批也可以",
        "可以不走审批",
        "帮你拆单",
        "可以拆单",
    ]

    if contains_any(normalized, unsupported_claims):
        return "边界回复包含未经 RAG 或工具支持的采购事实或审批结论"
    if contains_any(normalized, unrelated_fulfillment):
        return "边界回复实质性完成了无关任务"
    if contains_any(normalized, unsafe_compliance_phrases):
        return "边界回复包含规避采购制度或审批的表述"
    if "采购" not in normalized or not contains_any(normalized, ["预审", "合规", "需求", "申请"]):
        return "边界回复未明确引导回企业采购合规预审范围"

    return None


def contains_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def contains_unnegated_any(text: str, phrases: list[str]) -> bool:
    negation_markers = ["不", "无", "非", "未", "不能", "不可", "不得", "禁止", "避免", "暂缓", "暂停", "拒绝"]
    for phrase in phrases:
        start = text.find(phrase)
        while start != -1:
            prefix = text[max(0, start - 8) : start]
            if not any(marker in prefix for marker in negation_markers):
                return True
            start = text.find(phrase, start + 1)
    return False
