"""
Core Logic for Minh An B2B Sales AI Agent Workflow.
Integrates Gemini 3.8 Flash (via GeminiClient) with deterministic Pricing,
Inventory, Discount Policy, Approval Routing, and Pipeline Tracking.
"""
import json
import os
import re
from datetime import datetime
from gemini_client import GeminiClient


class MinhAnAgentSystem:
    def __init__(self, master_data_path="data_master.json", db_path="opportunities_db.json"):
        self.master_data_path = master_data_path
        self.db_path = db_path
        self.gemini = GeminiClient()
        self._load_master_data()
        self.opportunities = []
        self._load_db()

    def _load_master_data(self):
        with open(self.master_data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.price_list_version = self.data.get("price_list_version", "PL-2026-Q4")
        self.price_effective_date = self.data.get("price_effective_date", "2026-11-01")
        self.products = self.data["products"]
        self.sales_reps = self.data["sales_reps"]
        self.discount_policy = self.data["discount_policy"]

    def _save_master_data(self):
        with open(self.master_data_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def _load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self.opportunities = saved.get("opportunities", [])
                saved_counts = saved.get("rep_counts", {})
                for rep in self.sales_reps:
                    if rep["id"] in saved_counts:
                        rep["assigned_count"] = saved_counts[rep["id"]]
            except Exception:
                self.opportunities = []

    def _save_db(self):
        rep_counts = {r["id"]: r["assigned_count"] for r in self.sales_reps}
        payload = {
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rep_counts": rep_counts,
            "opportunities": self.opportunities,
        }
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def reset_demo_data(self):
        self.opportunities = []
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self._load_master_data()
        return {"status": "reset_ok"}

    def update_product_stock(self, sku, available_stock):
        for p in self.products:
            if p["sku"] == sku:
                p["available_stock"] = max(0, int(available_stock))
                self._save_master_data()
                return p
        raise ValueError(f"Không tìm thấy SKU: {sku}")

    def allocate_sales_rep(self):
        """Phân bổ NVKD có số lượng cơ hội đang xử lý ít nhất (Workload Balancing)."""
        sorted_reps = sorted(self.sales_reps, key=lambda r: r["assigned_count"])
        selected_rep = dict(sorted_reps[0])
        for r in self.sales_reps:
            if r["id"] == selected_rep["id"]:
                r["assigned_count"] += 1
                selected_rep["assigned_count"] = r["assigned_count"]
                break
        return selected_rep

    def _get_tier_max_discount(self, subtotal):
        tiers = sorted(self.discount_policy.get("volume_tiers", []), key=lambda t: t["min_total_vnd"])
        max_pct = 5.0
        for tier in tiers:
            if subtotal >= tier["min_total_vnd"]:
                max_pct = tier["max_discount_pct"]
        return max_pct

    # ------------------------------------------------------------------
    # BƯỚC 1 & 2: AI INTAKE & CLASSIFICATION AGENT
    # ------------------------------------------------------------------
    def parse_request_with_ai(self, text):
        """Sử dụng Gemini 3.8 Flash để bóc tách thực thể & ánh xạ danh mục SKU."""
        catalog_desc = [
            {
                "sku": p["sku"],
                "name": p["name"],
                "category": p["category"],
                "aliases": p.get("aliases", []),
                "unit_price_vnd": p["unit_price"],
            }
            for p in self.products
        ]

        system_instruction = (
            "Bạn là AI Intake Agent của Công ty Minh An (cung cấp thiết bị & dịch vụ công nghệ B2B). "
            "Nhiệm vụ của bạn là đọc văn bản yêu cầu từ email/biểu mẫu khách hàng và trích xuất dữ liệu chính xác tuyệt đối sang JSON. "
            "Tuyệt đối KHÔNG bỏ sót bất kỳ sản phẩm/dịch vụ nào khách nhắc tới: nếu khớp danh mục hãy đưa vào requested_items kèm đúng sku; "
            "nếu không khớp danh mục nào trong catalog hãy đưa vào unmatched_items để nhân viên kinh doanh không bỏ sót hạng mục."
        )

        prompt = f"""
Danh mục sản phẩm/dịch vụ hiện có của Minh An:
{json.dumps(catalog_desc, ensure_ascii=False, indent=2)}

Văn bản yêu cầu từ khách hàng:
\"\"\"
{text}
\"\"\"

Hãy trích xuất và trả về JSON đúng cấu trúc sau:
{{
  "customer_name": "Tên người liên hệ hoặc null",
  "company_name": "Tên công ty/tổ chức hoặc null",
  "email": "Email liên hệ hoặc null",
  "phone": "Số điện thoại liên hệ hoặc null",
  "customer_approver": "Tên/chức danh người phê duyệt phía khách hàng nếu có đề cập, hoặc null",
  "request_category": "Phân loại ngắn (VD: Hạ tầng Server & Mạng, Bảo mật, Dịch vụ triển khai, Tổng hợp)",
  "budget_vnd": "Số nguyên VND (VD: 300 triệu -> 300000000, 1.2 tỷ -> 1200000000, 250 củ -> 250000000) hoặc null nếu không đề cập",
  "budget_raw": "Chuỗi gốc nói về ngân sách hoặc null",
  "requested_deadline_days": "Số nguyên ngày làm việc (VD: 1 tuần -> 7, 5 ngày -> 5) hoặc null nếu không đề cập",
  "deadline_raw": "Chuỗi gốc nói về thời hạn hoặc null",
  "requested_discount_pct": "Số thực % chiết khấu khách đề nghị (mặc định 0.0 nếu không nói)",
  "requested_items": [
    {{
      "sku": "Mã SKU chính xác từ danh mục ở trên",
      "quantity": 1,
      "raw_mention": "Cụm từ khách hàng viết"
    }}
  ],
  "unmatched_items": [
    {{
      "raw_name": "Tên sản phẩm/dịch vụ khách yêu cầu nhưng KHÔNG có trong danh mục trên",
      "quantity": 1,
      "note": "Lý do chưa khớp danh mục"
    }}
  ]
}}
"""
        ai_data, meta = self.gemini.generate_json(prompt, system_instruction=system_instruction, timeout=25)
        return self._normalize_extracted(ai_data, meta)

    def parse_request_fallback(self, text, fallback_reason="Offline / Regex Fallback"):
        """Fallback phân tích bằng Regex khi chưa có API key hoặc mất mạng."""
        extracted = {
            "customer_name": None,
            "company_name": None,
            "email": None,
            "phone": None,
            "customer_approver": None,
            "request_category": "Thiết bị & Dịch vụ CNTT",
            "budget": None,
            "budget_raw": None,
            "requested_deadline_days": None,
            "deadline_raw": None,
            "requested_discount_pct": 0.0,
            "requested_items": [],
            "unmatched_items": [],
            "missing_fields": [],
            "advisory_notes": [],
            "ai_meta": {"provider": "regex-fallback", "model": "rule-engine", "used_ai": False, "latency_ms": 0, "error": fallback_reason},
        }

        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
        if email_match:
            extracted["email"] = email_match.group(0)

        phone_match = re.search(r"(0\d{9,10}|\+84\d{9,10})", text)
        if phone_match:
            extracted["phone"] = phone_match.group(0)

        budget_match = re.search(r"ngân sách[^0-9\n]*([0-9\.,]+)\s*(triệu|tr|củ|tỷ|vnd|đ)?", text, re.IGNORECASE)
        if budget_match:
            val_str = budget_match.group(1).replace(",", ".")
            if val_str.count(".") > 1:
                val_str = val_str.replace(".", "")
            unit = (budget_match.group(2) or "").lower()
            multiplier = 1
            if any(u in unit for u in ["triệu", "tr", "củ"]):
                multiplier = 1_000_000
            elif "tỷ" in unit:
                multiplier = 1_000_000_000
            try:
                val = float(val_str)
                if val < 10000 and multiplier == 1:
                    multiplier = 1_000_000
                extracted["budget"] = int(val * multiplier)
                extracted["budget_raw"] = budget_match.group(0)
            except ValueError:
                pass

        deadline_match = re.search(
            r"(?:thời hạn|tiến độ|giao trong|triển khai trong|hoàn thành trong|gấp trong)[^0-9\n]*([0-9]+)\s*(ngày|tuần)",
            text,
            re.IGNORECASE,
        )
        if deadline_match:
            num = int(deadline_match.group(1))
            unit = deadline_match.group(2).lower()
            extracted["requested_deadline_days"] = num * 7 if "tuần" in unit else num
            extracted["deadline_raw"] = deadline_match.group(0)

        discount_match = re.search(r"(?:chiết khấu|giảm giá)[^0-9\n]*([0-9]+(?:\.[0-9]+)?)\s*%", text, re.IGNORECASE)
        if discount_match:
            extracted["requested_discount_pct"] = float(discount_match.group(1))

        comp_match = re.search(r"(?:công ty|cty|doanh nghiệp)[:\s]*([^\n,\.\(]+)", text, re.IGNORECASE)
        if comp_match:
            extracted["company_name"] = comp_match.group(1).strip()

        contact_match = re.search(r"(?:tôi là|mình là|người liên hệ|tên)[:\s]*([A-ZĐÀ-Ỹa-zđà-ỹ\s]+)", text, re.IGNORECASE)
        if contact_match:
            extracted["customer_name"] = re.split(r"[\n,\-\(]", contact_match.group(1))[0].strip()

        approver_match = re.search(r"(?:phê duyệt bởi|người phê duyệt)[:\s]*([^\n,\.]+)", text, re.IGNORECASE)
        if approver_match:
            extracted["customer_approver"] = approver_match.group(1).strip()

        text_lower = text.lower()
        for prod in self.products:
            matched_kw = None
            for kw in prod.get("aliases", []):
                if kw in text_lower:
                    matched_kw = kw
                    break
            if matched_kw:
                qty = 1
                qty_match = re.search(
                    rf"\b([0-9]+)\s*(?:cái|bộ|chiếc|gói|con|máy|thiết bị)?\s*(?:[a-zđà-ỹ\s]{{0,25}})?{re.escape(matched_kw)}",
                    text_lower,
                )
                if not qty_match:
                    qty_match = re.search(rf"{re.escape(matched_kw)}[^\n,;]*?(?:sl|số lượng)[:\s]*([0-9]+)", text_lower)
                if qty_match:
                    try:
                        qty = max(1, int(qty_match.group(1)))
                    except ValueError:
                        qty = 1
                extracted["requested_items"].append({
                    "sku": prod["sku"],
                    "name": prod["name"],
                    "quantity": qty,
                    "raw_mention": matched_kw,
                })

        return self._validate_missing_fields(extracted)

    def _normalize_extracted(self, ai_data, meta):
        valid_skus = {p["sku"]: p for p in self.products}
        requested_items = []
        unmatched_items = []

        for item in ai_data.get("requested_items") or []:
            sku = (item.get("sku") or "").strip()
            qty = item.get("quantity", 1)
            try:
                qty = max(1, int(qty))
            except Exception:
                qty = 1
            if sku in valid_skus:
                requested_items.append({
                    "sku": sku,
                    "name": valid_skus[sku]["name"],
                    "quantity": qty,
                    "raw_mention": item.get("raw_mention") or valid_skus[sku]["name"],
                })
            else:
                unmatched_items.append({
                    "raw_name": item.get("raw_mention") or sku or "Hạng mục chưa rõ",
                    "quantity": qty,
                    "note": f"SKU '{sku}' không tồn tại trong danh mục hiện hành",
                })

        for um in ai_data.get("unmatched_items") or []:
            try:
                qty = max(1, int(um.get("quantity", 1)))
            except Exception:
                qty = 1
            unmatched_items.append({
                "raw_name": um.get("raw_name") or "Hạng mục ngoài danh mục",
                "quantity": qty,
                "note": um.get("note") or "Chưa có trong bảng giá chuẩn Minh An",
            })

        budget = ai_data.get("budget_vnd")
        try:
            budget = int(float(budget)) if budget is not None else None
        except Exception:
            budget = None

        deadline = ai_data.get("requested_deadline_days")
        try:
            deadline = int(float(deadline)) if deadline is not None else None
        except Exception:
            deadline = None

        discount = ai_data.get("requested_discount_pct")
        try:
            discount = float(discount) if discount is not None else 0.0
        except Exception:
            discount = 0.0

        extracted = {
            "customer_name": ai_data.get("customer_name"),
            "company_name": ai_data.get("company_name"),
            "email": ai_data.get("email"),
            "phone": ai_data.get("phone"),
            "customer_approver": ai_data.get("customer_approver"),
            "request_category": ai_data.get("request_category") or "Thiết bị & Dịch vụ CNTT",
            "budget": budget,
            "budget_raw": ai_data.get("budget_raw"),
            "requested_deadline_days": deadline,
            "deadline_raw": ai_data.get("deadline_raw"),
            "requested_discount_pct": discount,
            "requested_items": requested_items,
            "unmatched_items": unmatched_items,
            "missing_fields": [],
            "advisory_notes": [],
            "ai_meta": meta,
        }
        return self._validate_missing_fields(extracted)

    def _validate_missing_fields(self, extracted):
        missing = []
        if not extracted.get("email") and not extracted.get("phone"):
            missing.append("Thông tin liên hệ (Email hoặc Số điện thoại)")
        if not extracted.get("requested_items") and not extracted.get("unmatched_items"):
            missing.append("Danh sách sản phẩm/dịch vụ & số lượng cần báo giá")
        if not extracted.get("budget") or extracted["budget"] <= 0:
            missing.append("Ngân sách tổng dự kiến")
        if not extracted.get("requested_deadline_days") or extracted["requested_deadline_days"] <= 0:
            missing.append("Thời hạn / tiến độ triển khai mong muốn")
        if not extracted.get("customer_name") and not extracted.get("customer_approver"):
            missing.append("Thông tin người liên hệ / người phụ trách phê duyệt phía khách hàng")

        advisory = []
        if not extracted.get("customer_approver"):
            advisory.append("Chưa nêu rõ người phê duyệt ngân sách cuối cùng phía KH — NVKD cần xác nhận khi gửi đề xuất.")

        extracted["missing_fields"] = missing
        extracted["advisory_notes"] = advisory
        return extracted

    def parse_request(self, text):
        if self.gemini.is_configured():
            try:
                return self.parse_request_with_ai(text)
            except Exception as exc:
                return self.parse_request_fallback(text, fallback_reason=str(exc))
        return self.parse_request_fallback(text, fallback_reason="Chưa cấu hình GEMINI_API_KEY")

    # ------------------------------------------------------------------
    # SOẠN THẢO EMAIL BẰNG GEMINI 3.8 FLASH (CÓ FALLBACK TEMPLATE)
    # ------------------------------------------------------------------
    def _draft_clarification_email(self, parsed):
        if self.gemini.is_configured():
            try:
                prompt = f"""
Viết bản nháp email tiếng Việt chuyên nghiệp, lịch sự từ Bộ phận Kinh doanh - Công ty Cổ phần Công nghệ Minh An gửi cho khách hàng để yêu cầu làm rõ/bổ sung thông tin trước khi lập báo giá.

Thông tin đã nhận được:
- Tên người liên hệ: {parsed.get('customer_name') or 'Quý khách'}
- Công ty: {parsed.get('company_name') or 'Quý công ty'}
- Sản phẩm/dịch vụ đã đề cập: {json.dumps(parsed.get('requested_items', []), ensure_ascii=False)}
- Hạng mục ngoài danh mục (nếu có): {json.dumps(parsed.get('unmatched_items', []), ensure_ascii=False)}

Các thông tin còn THIẾU cần khách hàng bổ sung:
{json.dumps(parsed.get('missing_fields', []), ensure_ascii=False)}

Yêu cầu:
- Gồm Tiêu đề email (Subject:) và Nội dung email ngắn gọn, rõ ràng dạng gạch đầu dòng những thông tin còn thiếu.
- Không bịa đặt giá bán hoặc thời hạn giao hàng khi chưa đủ thông tin.
"""
                text, _ = self.gemini.generate_text(prompt, timeout=20)
                if text:
                    return text
            except Exception:
                pass

        lines = "\n".join(f" - {mf}" for mf in parsed["missing_fields"])
        return (
            f"Subject: [Minh An Tech] Xác nhận & bổ sung thông tin yêu cầu báo giá\n\n"
            f"Kính gửi Quý khách {parsed.get('customer_name') or ''} ({parsed.get('company_name') or 'Quý công ty'}),\n\n"
            f"Công ty Cổ phần Công nghệ Minh An chân thành cảm ơn sự quan tâm của Quý khách.\n"
            f"Để tránh sai sót hạng mục hoặc cam kết chưa sát nhu cầu, chúng tôi kính nhờ Quý khách bổ sung thêm các thông tin sau:\n"
            f"{lines}\n\n"
            f"Ngay khi nhận được phản hồi, chuyên viên kinh doanh Minh An sẽ kiểm tra tồn kho và gửi Đề xuất + Báo giá chính xác nhất.\n\n"
            f"Trân trọng,\nBộ phận Kinh doanh - Công ty Minh An."
        )

    def _draft_proposal_email(self, parsed, assigned_rep, evaluated_items, unmatched_items, financials, delivery, approval_info, manager_note=""):
        if self.gemini.is_configured():
            try:
                context = {
                    "customer_name": parsed.get("customer_name") or "Quý khách",
                    "company_name": parsed.get("company_name") or "Quý công ty",
                    "assigned_rep": assigned_rep,
                    "price_list_version": self.price_list_version,
                    "items": evaluated_items,
                    "unmatched_items": unmatched_items,
                    "financials": financials,
                    "delivery": delivery,
                    "approval_required": approval_info.get("requires_approval"),
                    "approval_reasons": approval_info.get("reasons"),
                    "manager_note": manager_note,
                }
                prompt = f"""
Hãy soạn một bản nháp Email phản hồi & Dự thảo Đề xuất Báo giá B2B bằng tiếng Việt chuyên nghiệp thay mặt cho nhân viên kinh doanh Công ty Cổ phần Công nghệ Minh An.

DỮ LIỆU THẨM ĐỊNH CHÍNH XÁC (Bắt buộc tuân thủ tuyệt đối 100% con số, KHÔNG tự ý thay đổi giá, số lượng, chiết khấu hay thời gian giao):
{json.dumps(context, ensure_ascii=False, indent=2)}

Quy tắc soạn thảo:
1. Có dòng Tiêu đề (Subject:).
2. Liệt kê ĐẦY ĐỦ tất cả hạng mục trong `items` (SKU, Tên, SL, Đơn giá, Thành tiền, Tình trạng tồn kho).
3. Nếu `unmatched_items` không rỗng, ghi rõ mục "Hạng mục cần Vận hành báo giá bổ sung" để tuyệt đối không bỏ sót yêu cầu của khách.
4. Trình bày rõ Tổng giá trị niêm yết, Chiết khấu ({financials['applied_discount_pct']}%), Tổng thanh toán dự kiến (Chưa VAT).
5. Trình bày trung thực thời gian giao hàng khả thi ({delivery['estimated_delivery_days']} ngày làm việc). Nếu `delivery.feasible` là false (không kịp tiến độ {delivery['requested_deadline_days']} ngày của khách), phải giải thích khéo léo nguyên nhân tồn kho/vận hành và đề xuất phương án xử lý (ví dụ giao trước phần hàng có sẵn).
6. Ký tên bằng đúng tên & email của `assigned_rep`.
"""
                text, _ = self.gemini.generate_text(prompt, timeout=25)
                if text:
                    return text
            except Exception:
                pass

        # Deterministic template fallback
        items_str = ""
        for i, it in enumerate(evaluated_items, 1):
            items_str += (
                f"{i}. [{it['sku']}] {it['name']} | SL: {it['quantity']} | "
                f"Đơn giá ({self.price_list_version}): {it['unit_price']:,} VNĐ | "
                f"Thành tiền: {it['total_price']:,} VNĐ ({it['stock_status']})\n"
            )
        unmatched_str = ""
        if unmatched_items:
            unmatched_str = "\n--- HẠNG MỤC CẦN BỔ SUNG BÁO GIÁ TỪ VẬN HÀNH ---\n"
            for u in unmatched_items:
                unmatched_str += f"- {u['raw_name']} (SL: {u['quantity']}) - {u['note']}\n"

        timeline_note = (
            f"Thời gian triển khai & giao hàng cam kết: {delivery['estimated_delivery_days']} ngày làm việc."
        )
        if not delivery["feasible"] and delivery["requested_deadline_days"]:
            timeline_note += (
                f" (Lưu ý: Tiến độ mong muốn {delivery['requested_deadline_days']} ngày hiện chưa đáp ứng ngay do thời gian chuẩn bị/nhập hàng là {delivery['estimated_delivery_days']} ngày; "
                f"chúng tôi đề xuất giao trước các thiết bị có sẵn trong kho)."
            )

        return (
            f"Subject: [Minh An Tech] Đề xuất giải pháp & Báo giá cho {parsed.get('company_name') or 'Quý công ty'}\n\n"
            f"Kính gửi Quý khách {parsed.get('customer_name') or 'đại diện'} - {parsed.get('company_name') or 'Quý công ty'},\n\n"
            f"Công ty Cổ phần Công nghệ Minh An xin gửi tới Quý khách Dự thảo Đề xuất & Báo giá theo bảng giá hiện hành ({self.price_list_version}):\n\n"
            f"--- CHI TIẾT HẠNG MỤC BÁO GIÁ ---\n"
            f"{items_str}{unmatched_str}\n"
            f"Tổng giá trị niêm yết: {financials['subtotal']:,.0f} VNĐ\n"
            f"Chiết khấu áp dụng ({financials['applied_discount_pct']}%): -{financials['discount_amount']:,.0f} VNĐ\n"
            f"TỔNG CỘNG THANH TOÁN (Chưa VAT): {financials['final_total']:,.0f} VNĐ\n"
            f"{timeline_note}\n\n"
            f"Chuyên viên phụ trách trực tiếp: {assigned_rep['name']} ({assigned_rep['email']})\n"
            f"Rất mong nhận được phản hồi của Quý công ty.\n\n"
            f"Trân trọng,\n{assigned_rep['name']} - Bộ phận Kinh doanh Minh An Company"
        )

    # ------------------------------------------------------------------
    # THẨM ĐỊNH TỒN KHO, GIÁ, CHIẾT KHẤU & THẨM QUYỀN
    # ------------------------------------------------------------------
    def _evaluate_and_build_result(self, opp_id, created_at, channel, raw_history, parsed, assigned_rep, workflow_logs, override_discount=None, manager_decision=None):
        evaluated_items = []
        subtotal = 0
        max_delivery_days = 0
        stock_alerts = []

        for item in parsed["requested_items"]:
            prod_info = next((p for p in self.products if p["sku"] == item["sku"]), None)
            if not prod_info:
                continue

            unit_price = prod_info["unit_price"]
            total_item_price = unit_price * item["quantity"]
            subtotal += total_item_price

            is_in_stock = prod_info["available_stock"] >= item["quantity"]
            lead_time = prod_info["standard_lead_time_days"] if is_in_stock else prod_info["backorder_lead_time_days"]
            if lead_time > max_delivery_days:
                max_delivery_days = lead_time

            item_status = (
                f"ĐỦ TỒN KHẢ DỤNG (Tồn: {prod_info['available_stock']})"
                if is_in_stock
                else f"THIẾU / HẾT KHO (Tồn: {prod_info['available_stock']})"
            )
            if not is_in_stock:
                stock_alerts.append(
                    f"'{prod_info['name']}' cần {item['quantity']} nhưng tồn khả dụng chỉ còn {prod_info['available_stock']}. "
                    f"Thời gian nhập bổ sung & giao hàng: {prod_info['backorder_lead_time_days']} ngày."
                )

            evaluated_items.append({
                "sku": prod_info["sku"],
                "name": prod_info["name"],
                "quantity": item["quantity"],
                "unit_price": unit_price,
                "total_price": total_item_price,
                "available_stock": prod_info["available_stock"],
                "in_stock": is_in_stock,
                "stock_status": item_status,
                "lead_time_days": lead_time,
            })

        req_discount = parsed["requested_discount_pct"]
        applied_discount = override_discount if override_discount is not None else req_discount
        tier_max_discount = self._get_tier_max_discount(subtotal)

        discount_amount = subtotal * (applied_discount / 100.0)
        final_total = subtotal - discount_amount

        budget = parsed.get("budget") or 0
        budget_gap = final_total - budget if budget else 0
        budget_overrun_pct = (budget_gap / budget * 100.0) if budget > 0 and budget_gap > 0 else 0.0

        req_deadline = parsed.get("requested_deadline_days")
        deadline_feasible = max_delivery_days <= (req_deadline or 999)

        # Kiểm tra điều kiện vượt thẩm quyền (nếu Trưởng phòng chưa xử lý)
        approval_reasons = []
        rep_limit = self.discount_policy["standard_rep_limit_pct"]
        if req_discount > rep_limit:
            approval_reasons.append(
                f"Chiết khấu đề nghị ({req_discount}%) vượt hạn mức NVKD tự duyệt (tối đa {rep_limit}%)."
            )
        if req_discount > tier_max_discount:
            approval_reasons.append(
                f"Chiết khấu đề nghị ({req_discount}%) vượt khung chính sách bậc doanh số (tối đa {tier_max_discount}% cho đơn {subtotal:,.0f} đ)."
            )
        if req_deadline and not deadline_feasible:
            approval_reasons.append(
                f"Khách hàng yêu cầu tiến độ {req_deadline} ngày, nhưng Vận hành cần tối thiểu {max_delivery_days} ngày."
            )
        if stock_alerts:
            approval_reasons.append(
                "Có sản phẩm hết/thiếu tồn kho cần Vận hành & Trưởng phòng xác nhận lịch nhập hàng trước khi cam kết."
            )
        if parsed.get("unmatched_items"):
            approval_reasons.append(
                f"Có {len(parsed['unmatched_items'])} hạng mục ngoài danh mục chuẩn cần Trưởng phòng & Vận hành định giá bổ sung."
            )
        if budget_overrun_pct > 10:
            approval_reasons.append(
                f"Tổng báo giá ({final_total:,.0f} đ) vượt ngân sách dự kiến của khách ({budget:,.0f} đ) là {budget_overrun_pct:.1f}%."
            )

        requires_manager = len(approval_reasons) > 0

        if manager_decision:
            if manager_decision["decision"] == "APPROVE":
                status = "APPROVED_TO_SEND"
            else:
                status = "REVISION_REQUIRED"
        else:
            status = "WAITING_MANAGER_APPROVAL" if requires_manager else "READY_TO_SEND"

        financials = {
            "price_list_version": self.price_list_version,
            "subtotal": subtotal,
            "requested_discount_pct": req_discount,
            "applied_discount_pct": applied_discount,
            "tier_max_discount_pct": tier_max_discount,
            "discount_amount": discount_amount,
            "final_total": final_total,
            "customer_budget": budget,
            "budget_gap": budget_gap,
            "budget_overrun_pct": round(budget_overrun_pct, 1),
        }

        delivery = {
            "requested_deadline_days": req_deadline,
            "estimated_delivery_days": max_delivery_days,
            "feasible": deadline_feasible,
            "stock_alerts": stock_alerts,
        }

        approval_info = {
            "requires_approval": requires_manager,
            "reasons": approval_reasons,
            "approver_role": self.discount_policy.get("manager_name", "Trưởng phòng Kinh doanh") if requires_manager else None,
            "manager_decision": manager_decision,
        }

        draft_email = self._draft_proposal_email(
            parsed=parsed,
            assigned_rep=assigned_rep,
            evaluated_items=evaluated_items,
            unmatched_items=parsed.get("unmatched_items", []),
            financials=financials,
            delivery=delivery,
            approval_info=approval_info,
            manager_note=manager_decision.get("note", "") if manager_decision else "",
        )

        return {
            "opportunity_id": opp_id,
            "created_at": created_at,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "channel": channel,
            "status": status,
            "tracking_stage": "Chờ Trưởng phòng duyệt" if status == "WAITING_MANAGER_APPROVAL" else ("Đã duyệt - Sẵn sàng gửi" if status in ("READY_TO_SEND", "APPROVED_TO_SEND") else "Cần NVKD điều chỉnh"),
            "raw_history": raw_history,
            "parsed_data": parsed,
            "assigned_rep": assigned_rep,
            "items": evaluated_items,
            "unmatched_items": parsed.get("unmatched_items", []),
            "financials": financials,
            "delivery": delivery,
            "approval_info": approval_info,
            "draft_response": draft_email,
            "workflow_logs": workflow_logs,
        }

    # ------------------------------------------------------------------
    # PUBLIC WORKFLOW ENTRYPOINTS
    # ------------------------------------------------------------------
    def process_opportunity(self, input_text, channel="Email"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        opp_id = f"OPP-{datetime.now().strftime('%y%m%d')}-{len(self.opportunities) + 1:03d}"

        parsed = self.parse_request(input_text)
        raw_history = [{"time": timestamp, "sender": "Khách hàng", "channel": channel, "content": input_text}]
        workflow_logs = [
            {"time": timestamp, "step": "1. Tiếp nhận yêu cầu", "detail": f"Nhận yêu cầu qua {channel}"},
            {
                "time": timestamp,
                "step": "2. Phân loại & Kiểm tra thông tin",
                "detail": f"AI ({parsed['ai_meta']['model']}) trích xuất xong. Thiếu {len(parsed['missing_fields'])} trường.",
            },
        ]

        # Nhánh KHÔNG ĐỦ THÔNG TIN -> Yêu cầu KH làm rõ / bổ sung
        if parsed["missing_fields"]:
            draft_clarification = self._draft_clarification_email(parsed)
            workflow_logs.append({
                "time": timestamp,
                "step": "Nhánh rẽ: Yêu cầu KH làm rõ / bổ sung",
                "detail": "Thiếu: " + ", ".join(parsed["missing_fields"]),
            })
            result = {
                "opportunity_id": opp_id,
                "created_at": timestamp,
                "updated_at": timestamp,
                "channel": channel,
                "status": "NEED_MORE_INFO",
                "tracking_stage": "Chờ KH bổ sung thông tin",
                "raw_history": raw_history,
                "parsed_data": parsed,
                "missing_fields": parsed["missing_fields"],
                "assigned_rep": None,
                "items": [],
                "unmatched_items": parsed.get("unmatched_items", []),
                "financials": None,
                "delivery": None,
                "approval_info": None,
                "draft_response": draft_clarification,
                "workflow_logs": workflow_logs,
            }
            self.opportunities.append(result)
            self._save_db()
            return result

        # Đủ thông tin -> Tạo Hồ sơ Cơ hội & Gán người phụ trách
        assigned_rep = self.allocate_sales_rep()
        workflow_logs.append({
            "time": timestamp,
            "step": "3. Tạo Hồ sơ & Gán NVKD",
            "detail": f"Gán phụ trách: {assigned_rep['name']} ({assigned_rep['id']})",
        })
        workflow_logs.append({
            "time": timestamp,
            "step": "4. Thẩm định Tồn kho & Giá",
            "detail": f"Đối chiếu bảng giá {self.price_list_version} và tồn kho Vận hành.",
        })

        result = self._evaluate_and_build_result(
            opp_id=opp_id,
            created_at=timestamp,
            channel=channel,
            raw_history=raw_history,
            parsed=parsed,
            assigned_rep=assigned_rep,
            workflow_logs=workflow_logs,
        )

        if result["status"] == "WAITING_MANAGER_APPROVAL":
            result["workflow_logs"].append({
                "time": timestamp,
                "step": "5. Kiểm tra Thẩm quyền",
                "detail": "Vượt thẩm quyền -> Chuyển Trưởng phòng Kinh doanh duyệt.",
            })
        else:
            result["workflow_logs"].append({
                "time": timestamp,
                "step": "5. Hoàn tất Dự thảo Đề xuất",
                "detail": "Trong thẩm quyền NVKD -> Sẵn sàng gửi khách hàng.",
            })

        self.opportunities.append(result)
        self._save_db()
        return result

    def clarify_opportunity(self, opp_id, reply_text):
        """Vòng lặp quy trình: KH phản hồi bổ sung -> quay lại Tiếp nhận & Kiểm tra thông tin."""
        idx = next((i for i, o in enumerate(self.opportunities) if o["opportunity_id"] == opp_id), None)
        if idx is None:
            raise ValueError(f"Không tìm thấy cơ hội {opp_id}")

        opp = self.opportunities[idx]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        raw_history = opp.get("raw_history", [])
        raw_history.append({"time": timestamp, "sender": "Khách hàng (Bổ sung)", "channel": opp.get("channel", "Email"), "content": reply_text})

        combined_text = "\n\n--- THÔNG TIN BỔ SUNG TỪ KHÁCH HÀNG ---\n\n".join(h["content"] for h in raw_history)
        parsed = self.parse_request(combined_text)

        logs = opp.get("workflow_logs", [])
        logs.append({
            "time": timestamp,
            "step": "Vòng lặp: Nhận phản hồi bổ sung từ KH",
            "detail": f"Tái kiểm tra thông tin. Còn thiếu {len(parsed['missing_fields'])} trường.",
        })

        if parsed["missing_fields"]:
            opp["updated_at"] = timestamp
            opp["raw_history"] = raw_history
            opp["parsed_data"] = parsed
            opp["missing_fields"] = parsed["missing_fields"]
            opp["unmatched_items"] = parsed.get("unmatched_items", [])
            opp["draft_response"] = self._draft_clarification_email(parsed)
            opp["workflow_logs"] = logs
            self._save_db()
            return opp

        assigned_rep = opp.get("assigned_rep") or self.allocate_sales_rep()
        logs.append({
            "time": timestamp,
            "step": "3. Tạo Hồ sơ & Gán NVKD",
            "detail": f"Đã đủ thông tin! Gán phụ trách: {assigned_rep['name']} ({assigned_rep['id']})",
        })

        updated_opp = self._evaluate_and_build_result(
            opp_id=opp["opportunity_id"],
            created_at=opp["created_at"],
            channel=opp.get("channel", "Email"),
            raw_history=raw_history,
            parsed=parsed,
            assigned_rep=assigned_rep,
            workflow_logs=logs,
        )
        self.opportunities[idx] = updated_opp
        self._save_db()
        return updated_opp

    def manager_decision(self, opp_id, decision, approved_discount_pct=None, note=""):
        """Trưởng phòng Kinh doanh Phê duyệt hoặc Yêu cầu điều chỉnh."""
        idx = next((i for i, o in enumerate(self.opportunities) if o["opportunity_id"] == opp_id), None)
        if idx is None:
            raise ValueError(f"Không tìm thấy cơ hội {opp_id}")

        opp = self.opportunities[idx]
        if opp["status"] == "NEED_MORE_INFO":
            raise ValueError("Hồ sơ chưa đủ thông tin để duyệt.")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        decision_upper = decision.upper()
        if decision_upper not in ("APPROVE", "REJECT"):
            raise ValueError("Decision phải là APPROVE hoặc REJECT")

        override_disc = None
        if approved_discount_pct is not None and approved_discount_pct != "":
            override_disc = float(approved_discount_pct)

        mgr_info = {
            "decision": decision_upper,
            "approved_discount_pct": override_disc if override_disc is not None else opp["financials"]["requested_discount_pct"],
            "note": note,
            "decided_at": timestamp,
            "approver": self.discount_policy.get("manager_name", "Trưởng phòng Kinh doanh"),
        }

        logs = opp.get("workflow_logs", [])
        logs.append({
            "time": timestamp,
            "step": "6. Trưởng phòng duyệt",
            "detail": f"{'PHÊ DUYỆT' if decision_upper == 'APPROVE' else 'YÊU CẦU ĐIỀU CHỈNH'} (CK duyệt: {mgr_info['approved_discount_pct']}%). Ghi chú: {note or 'Không có'}",
        })

        updated_opp = self._evaluate_and_build_result(
            opp_id=opp["opportunity_id"],
            created_at=opp["created_at"],
            channel=opp.get("channel", "Email"),
            raw_history=opp.get("raw_history", []),
            parsed=opp["parsed_data"],
            assigned_rep=opp["assigned_rep"],
            workflow_logs=logs,
            override_discount=mgr_info["approved_discount_pct"],
            manager_decision=mgr_info,
        )
        self.opportunities[idx] = updated_opp
        self._save_db()
        return updated_opp

    def update_tracking(self, opp_id, tracking_stage, note=""):
        """Bước cuối: Cập nhật & Theo dõi tiến độ xử lý cơ hội."""
        opp = next((o for o in self.opportunities if o["opportunity_id"] == opp_id), None)
        if not opp:
            raise ValueError(f"Không tìm thấy cơ hội {opp_id}")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        opp["tracking_stage"] = tracking_stage
        opp["updated_at"] = timestamp
        opp.setdefault("workflow_logs", []).append({
            "time": timestamp,
            "step": "7. Cập nhật & Theo dõi",
            "detail": f"Chuyển trạng thái theo dõi -> '{tracking_stage}'. {note}".strip(),
        })
        self._save_db()
        return opp

    def get_dashboard_summary(self):
        total = len(self.opportunities)
        need_info = sum(1 for o in self.opportunities if o["status"] == "NEED_MORE_INFO")
        waiting_mgr = sum(1 for o in self.opportunities if o["status"] == "WAITING_MANAGER_APPROVAL")
        ready = sum(1 for o in self.opportunities if o["status"] in ("READY_TO_SEND", "APPROVED_TO_SEND"))
        revision = sum(1 for o in self.opportunities if o["status"] == "REVISION_REQUIRED")
        return {
            "total": total,
            "need_info": need_info,
            "waiting_manager": waiting_mgr,
            "ready": ready,
            "revision": revision,
            "ai_configured": self.gemini.is_configured(),
            "price_list_version": self.price_list_version,
            "price_effective_date": self.price_effective_date,
        }
