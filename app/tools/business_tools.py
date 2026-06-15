from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.models import PurchaseRequest


class BusinessTools:
    def __init__(self, data_path: Path | None = None) -> None:
        self.data_path = data_path or Path(__file__).resolve().parents[1] / "data" / "business.json"
        self.data: dict[str, Any] = json.loads(self.data_path.read_text(encoding="utf-8"))

    def query_budget(self, purchase: PurchaseRequest) -> dict[str, Any]:
        department = purchase.department
        budget_category = purchase.budget_category
        amount = purchase.amount
        if not department or not budget_category or amount is None:
            return {
                "ok": False,
                "risk_level": "HIGH",
                "message": "预算查询缺少部门、预算科目或金额参数。",
            }

        department_budgets = self.data["budgets"].get(department, {})
        budget = department_budgets.get(budget_category)
        if not budget:
            return {
                "ok": True,
                "department": department,
                "budget_category": budget_category,
                "budget_category_confirmed": purchase.budget_category_confirmed,
                "available_budget": 0,
                "requested_amount": amount,
                "is_sufficient": False,
                "category_matched": False,
                "risk_level": "HIGH",
                "owner": "预算负责人",
                "message": f"{department} 未找到 {budget_category} 对应预算科目。",
            }

        available = float(budget["available_budget"])
        sufficient = available >= amount
        return {
            "ok": True,
            "department": department,
            "budget_category": budget_category,
            "budget_category_confirmed": purchase.budget_category_confirmed,
            "available_budget": available,
            "requested_amount": amount,
            "is_sufficient": sufficient,
            "category_matched": True,
            "risk_level": "LOW" if sufficient else "HIGH",
            "owner": budget["owner"],
            "message": (
                f"{department}{budget_category}预算余额 {available:.0f} 元，本次申请 {amount:.0f} 元，"
                f"{'预算充足' if sufficient else '预算不足'}。"
                + ("" if purchase.budget_category_confirmed else "预算科目为 Agent 推断，提交前需用户确认。")
            ),
        }

    def query_vendor_qualification(self, purchase: PurchaseRequest) -> dict[str, Any]:
        vendor_name = purchase.vendor_name
        if not vendor_name:
            return {"ok": False, "risk_level": "LOW", "message": "未指定供应商，无需查询供应商资质。"}

        vendor = self.data["vendors"].get(vendor_name)
        if not vendor:
            return {
                "ok": True,
                "vendor_name": vendor_name,
                "is_approved": False,
                "qualification_status": "未查询到准入记录",
                "approved_categories": [],
                "last_review_date": None,
                "risk_level": "HIGH",
                "message": f"未查询到 {vendor_name} 的准入资质记录。",
            }

        category_matched = (
            not purchase.purchase_category
            or purchase.purchase_category in vendor["approved_categories"]
            or (purchase.purchase_category == "直播设备" and "运营设备" in vendor["approved_categories"])
        )
        risk_level = "LOW" if vendor["is_approved"] and vendor["qualification_status"] == "有效" and category_matched else "HIGH"
        return {
            "ok": True,
            "vendor_name": vendor_name,
            "is_approved": vendor["is_approved"],
            "qualification_status": vendor["qualification_status"],
            "approved_categories": vendor["approved_categories"],
            "category_matched": category_matched,
            "last_review_date": vendor["last_review_date"],
            "risk_level": risk_level,
            "message": f"{vendor_name} {'已准入且资质有效' if risk_level == 'LOW' else '存在准入或品类匹配风险'}。",
        }

    def query_vendor_risk(self, purchase: PurchaseRequest) -> dict[str, Any]:
        vendor_name = purchase.vendor_name
        if not vendor_name:
            return {"ok": False, "risk_level": "LOW", "message": "未指定供应商，无需查询供应商风险。"}

        vendor = self.data["vendors"].get(vendor_name)
        if not vendor:
            return {
                "ok": True,
                "vendor_name": vendor_name,
                "blacklist_hit": False,
                "watchlist_hit": True,
                "abnormal_records": ["未查询到供应商风险档案"],
                "complaint_count": 0,
                "risk_level": "HIGH",
                "message": f"未查询到 {vendor_name} 的完整风险档案，需人工复核。",
            }

        risk = vendor["risk"]
        return {
            "ok": True,
            "vendor_name": vendor_name,
            "blacklist_hit": risk["blacklist_hit"],
            "watchlist_hit": risk["watchlist_hit"],
            "abnormal_records": risk["abnormal_records"],
            "complaint_count": risk["complaint_count"],
            "risk_level": risk["risk_level"],
            "message": f"{vendor_name} 风险等级为 {risk['risk_level']}，异常记录 {len(risk['abnormal_records'])} 条，投诉 {risk['complaint_count']} 次。",
        }

    def query_historical_price(self, purchase: PurchaseRequest) -> dict[str, Any]:
        item_name = purchase.item_name or purchase.purchase_category
        if not item_name or purchase.amount is None or not purchase.quantity:
            return {
                "ok": False,
                "risk_level": "MEDIUM",
                "message": "历史价格查询缺少商品、金额或数量，无法计算预算单价。",
            }

        price = self.data["historical_prices"].get(item_name)
        if not price:
            return {
                "ok": True,
                "item_name": item_name,
                "sample_size": 0,
                "risk_level": "MEDIUM",
                "message": f"未找到 {item_name} 的历史价格样本，价格风险依据不足。",
            }

        requested_unit_price = purchase.amount / purchase.quantity
        avg = float(price["historical_avg_unit_price"])
        deviation_rate = ((requested_unit_price - avg) / avg) * 100
        if deviation_rate > 20:
            risk_level = "HIGH"
        elif deviation_rate > 5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "ok": True,
            "item_name": item_name,
            "historical_avg_unit_price": avg,
            "historical_min_unit_price": price["historical_min_unit_price"],
            "historical_max_unit_price": price["historical_max_unit_price"],
            "requested_unit_price": round(requested_unit_price, 2),
            "deviation_rate": round(deviation_rate, 1),
            "sample_size": price["sample_size"],
            "risk_level": risk_level,
            "message": f"{item_name} 本次预算单价 {requested_unit_price:.0f} 元，较历史均价偏离 {deviation_rate:.1f}%。",
        }

    def query_approval_chain(self, purchase: PurchaseRequest, risk_types: list[str]) -> dict[str, Any]:
        amount = purchase.amount or 0
        approvers: list[str] = []
        reasons: list[str] = []

        if amount >= 100000:
            approvers.extend(["采购专员", "部门负责人"])
            reasons.append("金额达到 10 万元及以上，需采购复核和部门负责人确认")
        elif amount > 0:
            approvers.append("部门负责人")
            reasons.append("常规采购需部门负责人确认")

        if purchase.purchase_category == "IT设备":
            approvers.append("IT 负责人")
            reasons.append("IT 设备采购需 IT 部门确认型号和配置")

        if any(risk in risk_types for risk in ["VENDOR_BLACKLIST_RISK", "VENDOR_QUALIFICATION_RISK"]):
            approvers.extend(["采购专员", "合规人员"])
            reasons.append("供应商存在准入或异常风险")

        if "BUDGET_RISK" in risk_types:
            approvers.append("预算负责人")
            reasons.append("预算余额或预算科目存在风险")

        unique_approvers = list(dict.fromkeys(approvers))
        approval_required = bool(unique_approvers)
        level = "高级采购审批" if amount >= 500000 else "采购复核" if amount >= 100000 else "部门审批"
        return {
            "ok": True,
            "approval_required": approval_required,
            "approvers": unique_approvers,
            "approval_level": level,
            "reason": "；".join(reasons) if reasons else "未触发额外审批链。",
            "message": f"建议审批链：{', '.join(unique_approvers) if unique_approvers else '无需额外人工审批'}。",
        }
