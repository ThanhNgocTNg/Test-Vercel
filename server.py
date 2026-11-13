"""
Web Server for Minh An B2B Sales AI Agent Workflow.
Runs on localhost:8000 with multi-threaded HTTP handling.
"""
import http.server
import json
import socketserver
import urllib.parse
from agent_core import MinhAnAgentSystem

PORT = 8000
agent_system = MinhAnAgentSystem()


class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class MinhAnHandler(http.server.SimpleHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        raw = self.rfile.read(content_length).decode("utf-8")
        return json.loads(raw)

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/" or path == "/index.html":
            with open("index.html", "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if path == "/api/status":
            self._send_json(agent_system.get_dashboard_summary())
            return

        if path == "/api/reps":
            self._send_json(agent_system.sales_reps)
            return

        if path == "/api/opportunities":
            self._send_json(agent_system.opportunities)
            return

        if path == "/api/products":
            self._send_json({
                "price_list_version": agent_system.price_list_version,
                "price_effective_date": agent_system.price_effective_date,
                "products": agent_system.products,
                "discount_policy": agent_system.discount_policy,
            })
            return

        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        try:
            data = self._read_json_body()

            if path == "/api/process":
                raw_text = (data.get("text") or "").strip()
                if not raw_text:
                    self._send_json({"error": "Vui lòng nhập nội dung yêu cầu"}, status=400)
                    return
                channel = data.get("channel") or "Email"
                result = agent_system.process_opportunity(raw_text, channel=channel)
                self._send_json(result)
                return

            if path == "/api/clarify":
                opp_id = data.get("opportunity_id", "")
                reply_text = (data.get("reply_text") or "").strip()
                if not opp_id or not reply_text:
                    self._send_json({"error": "Thiếu opportunity_id hoặc nội dung phản hồi"}, status=400)
                    return
                result = agent_system.clarify_opportunity(opp_id, reply_text)
                self._send_json(result)
                return

            if path == "/api/approve":
                opp_id = data.get("opportunity_id", "")
                decision = data.get("decision", "")
                approved_discount = data.get("approved_discount_pct")
                note = data.get("note", "")
                result = agent_system.manager_decision(
                    opp_id, decision, approved_discount_pct=approved_discount, note=note
                )
                self._send_json(result)
                return

            if path == "/api/track":
                opp_id = data.get("opportunity_id", "")
                stage = data.get("tracking_stage", "")
                note = data.get("note", "")
                result = agent_system.update_tracking(opp_id, stage, note=note)
                self._send_json(result)
                return

            if path == "/api/stock":
                sku = data.get("sku", "")
                stock = data.get("available_stock", 0)
                updated = agent_system.update_product_stock(sku, stock)
                self._send_json(updated)
                return

            if path == "/api/reset":
                self._send_json(agent_system.reset_demo_data())
                return

            self._send_json({"error": "Endpoint không tồn tại"}, status=404)

        except Exception as e:
            self._send_json({"error": str(e)}, status=500)


if __name__ == "__main__":
    with ThreadingHTTPServer(("127.0.0.1", PORT), MinhAnHandler) as httpd:
        ai_state = "Gemini API Ready" if agent_system.gemini.is_configured() else "Fallback Mode (No Key)"
        print("=" * 64)
        print(f"🚀 Minh An B2B Sales AI Agent chạy tại: http://localhost:{PORT}")
        print(f"🤖 AI Engine: {ai_state} | Bảng giá: {agent_system.price_list_version}")
        print("=" * 64)
        httpd.serve_forever()
