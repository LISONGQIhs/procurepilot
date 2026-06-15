from __future__ import annotations

import re

from app.models import PurchaseRequest


KNOWN_DEPARTMENTS = ["行政部", "市场部", "运营部", "IT部", "财务部", "人力资源部", "销售部", "法务部", "采购部"]

ITEM_RULES: list[tuple[str, str, str]] = [
    ("MacBook", "IT设备", "MacBook"),
    ("笔记本", "IT设备", "笔记本电脑"),
    ("电脑", "IT设备", "电脑"),
    ("显示器", "IT设备", "显示器"),
    ("服务器", "IT设备", "服务器"),
    ("办公椅", "办公用品", "办公椅"),
    ("椅", "办公用品", "办公椅"),
    ("文具", "办公用品", "文具"),
    ("耗材", "办公用品", "办公耗材"),
    ("直播设备", "直播设备", "直播设备"),
    ("摄像机", "直播设备", "摄像机"),
]

KNOWN_VENDORS = ["示例供应商A", "示例供应商B", "示例供应商C", "未准入供应商A"]


def parse_amount(text: str) -> float | None:
    patterns = [
        r"(?:预算|金额|总价|总金额|申请金额|大概|约|预计|本次申请)?\s*([0-9]+(?:\.[0-9]+)?)\s*(万元|万|元)",
        r"([0-9]+(?:\.[0-9]+)?)\s*(万元|万|元)\s*(?:以下|左右|以内)?",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            number = float(match.group(1))
            unit = match.group(2)
            if unit in {"万元", "万"}:
                return number * 10000
            return number
    return None


def parse_quantity(text: str) -> int | None:
    match = re.search(r"([0-9]+)\s*(?:台|把|个|件|套|批|组)", text)
    if not match:
        return None
    return int(match.group(1))


def parse_vendor(text: str) -> str | None:
    for vendor in KNOWN_VENDORS:
        if vendor in text:
            return vendor

    patterns = [
        r"供应商(?:是|为)?\s*([\u4e00-\u9fa5A-Za-z0-9]+)",
        r"指定\s*([\u4e00-\u9fa5A-Za-z0-9]+?)\s*(?:供货|供应)",
        r"找\s*([\u4e00-\u9fa5A-Za-z0-9]+?)\s*(?:供货|供应)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip(" ，。,.")
    return None


def parse_budget_category(text: str) -> str | None:
    match = re.search(r"预算科目(?:是|为|：|:)?\s*([\u4e00-\u9fa5A-Za-z0-9]+)", text)
    if match:
        return match.group(1).strip(" ，。,.")

    known_categories = ["行政办公", "活动运营", "运营设备", "IT设备", "客户活动"]
    for category in known_categories:
        if f"预算科目{category}" in text or f"{category}预算" in text:
            return category
    return None


def parse_purpose(text: str) -> str | None:
    match = re.search(r"用于([^。；;，,\n]+)", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"给([^。；;，,\n]+?)(?:补充|使用|用)", text)
    if match:
        return match.group(1).strip()
    return None


def parse_delivery(text: str) -> str | None:
    patterns = [
        r"两周内",
        r"一周内",
        r"本周内",
        r"月底前",
        r"尽快到货",
        r"尽快",
        r"[0-9]+\s*天内",
        r"[0-9]+\s*周内",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None


def infer_budget_category(department: str | None, category: str | None, purpose: str | None) -> tuple[str | None, str | None]:
    purpose = purpose or ""
    if "活动" in purpose or "运营" in purpose:
        return "活动运营" if department == "市场部" else "运营设备", "Agent 根据用途推断，需用户确认"
    if category == "办公用品":
        return "行政办公", "Agent 根据品类推断，需用户确认"
    if category == "IT设备":
        return "IT设备", "Agent 根据品类推断，需用户确认"
    if category == "直播设备":
        return "运营设备", "Agent 根据品类推断，需用户确认"
    return None, None


def extract_purchase_request(text: str) -> PurchaseRequest:
    source_fields: dict[str, str] = {}
    confidence: dict[str, float] = {}
    flags: list[str] = []

    department = next((dept for dept in KNOWN_DEPARTMENTS if dept in text), None)
    if department:
        source_fields["department"] = "用户原始输入"
        confidence["department"] = 0.95
    elif "我们部门" in text or "本部门" in text:
        flags.append("DEPARTMENT_AMBIGUOUS")

    purchase_category = None
    item_name = None
    for keyword, category, item in ITEM_RULES:
        if keyword in text:
            purchase_category = category
            item_name = item
            source_fields["purchase_category"] = "用户原始输入"
            source_fields["item_name"] = "用户原始输入"
            confidence["purchase_category"] = 0.9
            confidence["item_name"] = 0.9
            break

    quantity = parse_quantity(text)
    if quantity is not None:
        source_fields["quantity"] = "用户原始输入"
        confidence["quantity"] = 0.9
    elif "一批" in text or "若干" in text:
        flags.append("QUANTITY_AMBIGUOUS")

    amount = parse_amount(text)
    if amount is not None:
        source_fields["amount"] = "用户原始输入"
        confidence["amount"] = 0.95

    purpose = parse_purpose(text)
    if purpose:
        source_fields["purpose"] = "用户原始输入"
        confidence["purpose"] = 0.85

    vendor_name = parse_vendor(text)
    if vendor_name:
        source_fields["vendor_name"] = "用户原始输入"
        confidence["vendor_name"] = 0.9

    delivery_requirement = parse_delivery(text)
    if delivery_requirement:
        source_fields["delivery_requirement"] = "用户原始输入"
        confidence["delivery_requirement"] = 0.8

    budget_category = parse_budget_category(text)
    budget_category_confirmed = False
    if budget_category:
        budget_category_confirmed = True
        source_fields["budget_category"] = "用户原始输入"
        confidence["budget_category"] = 0.95
    else:
        budget_category, budget_source = infer_budget_category(department, purchase_category, purpose)
        if budget_category and budget_source:
            source_fields["budget_category"] = budget_source
            confidence["budget_category"] = 0.65

    is_urgent = any(word in text for word in ["尽快", "紧急", "急需", "马上"]) or bool(
        delivery_requirement and any(word in delivery_requirement for word in ["周内", "天内", "到货"])
    )
    specified_brand_or_model = any(word in text for word in ["指定", "MacBook", "指定品牌", "指定型号"])

    if any(word in text for word in ["拆单", "拆成", "拆分", "不用走", "绕过审批", "规避审批"]):
        flags.append("USER_COMPLIANCE_RISK")

    if "预算" in text and "总价" in text:
        flags.append("POSSIBLE_AMOUNT_CONFLICT")

    return PurchaseRequest(
        department=department,
        purchase_category=purchase_category,
        item_name=item_name,
        quantity=quantity,
        amount=amount,
        purpose=purpose,
        vendor_name=vendor_name,
        delivery_requirement=delivery_requirement,
        budget_category=budget_category,
        budget_category_confirmed=budget_category_confirmed,
        is_urgent=is_urgent,
        specified_brand_or_model=specified_brand_or_model,
        source_fields=source_fields,
        confidence=confidence,
        flags=flags,
    )
