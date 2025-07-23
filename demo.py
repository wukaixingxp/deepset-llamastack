import logging
import os
import re
import threading
import time
from urllib.parse import urlparse

import gradio as gr
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack_integrations.components.generators.llama_stack.chat.chat_generator import (
    LlamaStackChatGenerator,
)

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


class ChatInterface:
    def __init__(self):
        self.client = None
        self.messages = []
        self.streaming_response = ""
        self.is_streaming = False
        self.lock = threading.Lock()

    def validate_inputs(self, url, model):

        if not model or not model.strip():
            return False, "Model name cannot be empty"

        url = url.strip() or "http://0.0.0.0:8321/v1/openai"
        if not url.startswith(("http://", "https://")):
            url = "http://" + url

        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("Invalid URL structure")

            # Basic IP/hostname validation
            netloc = parsed.netloc.split(":")[0]
            if not (
                netloc in ["localhost", "0.0.0.0"]
                or any(
                    netloc.startswith(prefix) for prefix in ["127.", "192.168.", "10."]
                )
                or re.match(r"^(\d{1,3}\.){3}\d{1,3}$", netloc)
                or "." in netloc
            ):
                raise ValueError("Invalid hostname or IP address")

        except Exception as e:
            return False, f"Invalid URL: {str(e)}"

        if not url.endswith("/v1"):
            url = url.rstrip("/") + "/v1"
        return True, url

    def initialize_client(self, url, model):
        valid, result = self.validate_inputs(url, model)
        if not valid:
            return False, result

        try:
            self.client = LlamaStackChatGenerator(
                api_base_url=result,
                model=model.strip(),
                streaming_callback=self.streaming_callback,
            )
            return True, f"Connected to {result} with model '{model.strip()}'"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def streaming_callback(self, chunk):
        with self.lock:
            content = getattr(chunk, "content", None) or getattr(chunk, "text", "")
            if content:
                self.streaming_response += content

    def chat(self, message, history, url, model):
        history = history or []

        if not self.client:
            success, msg = self.initialize_client(url, model)
            if not success:
                yield history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": f"❌ {msg}"},
                ]
                return

        if not message or not message.strip():
            yield history
            return

        # Add user message and reset streaming state
        new_history = history + [{"role": "user", "content": message}]
        yield new_history

        self.messages.append(ChatMessage.from_user(message.strip()))

        with self.lock:
            self.streaming_response = ""
            self.is_streaming = True

        # Stream response in background thread
        def stream_response():
            try:
                response = self.client.run(messages=self.messages)["replies"][0]
                final_response = getattr(response, "text", str(response))
                self.messages.append(ChatMessage.from_assistant(final_response))

                with self.lock:
                    if not self.streaming_response:
                        self.streaming_response = final_response
                    self.is_streaming = False
            except Exception as e:
                with self.lock:
                    self.streaming_response = f"❌ Error: {str(e)}"
                    self.is_streaming = False

        thread = threading.Thread(target=stream_response, daemon=True)
        thread.start()

        # Stream updates to UI
        last_response = ""
        while self.is_streaming or thread.is_alive():
            with self.lock:
                current_response = self.streaming_response

            if current_response != last_response:
                current_history = new_history + [
                    {"role": "assistant", "content": current_response}
                ]
                yield current_history
                last_response = current_response

            time.sleep(0.05)

        with self.lock:
            if self.streaming_response:
                final_history = new_history + [
                    {"role": "assistant", "content": self.streaming_response}
                ]
                yield final_history

    def clear_chat(self):
        """Clear chat history"""
        with self.lock:
            self.messages = []
            self.streaming_response = ""
            self.is_streaming = False
        return []  # Empty list is compatible with both formats


# Create interface instance
chat_interface = ChatInterface()

CUSTOM_CSS = """
.gradio-container { background-color: #ffffff !important; color: #000000 !important; }
.gr-textbox { background-color: #ffffff !important; border: 1px solid #e5e5e5 !important; color: #000000 !important; border-radius: 8px !important; }
.gr-textbox input { background-color: #ffffff !important; color: #000000 !important; }
.gr-textbox input::placeholder { color: #6e6e80 !important; }
.gr-button { background-color: #10a37f !important; color: #ffffff !important; border: none !important; border-radius: 4px !important; }
.gr-button:hover { background-color: #0e8e6d !important; }
.gr-chatbot { background-color: #ffffff !important; border: 1px solid #e5e5e5 !important; border-radius: 8px !important; }
.gr-chatbot .message { padding: 16px !important; }
.gr-chatbot .user { background-color: #f7f7f8 !important; }
.gr-chatbot .bot { background-color: #ffffff !important; }
.status-indicator { background-color: #ffffff !important; border: 1px solid #e5e5e5 !important; padding: 12px !important; color: #000000 !important; border-radius: 8px !important; }
.status-success { background-color: #e9f9f2 !important; color: #10a37f !important; }
.error-message { background-color: #fef2f2 !important; color: #ef4444 !important; }
.header-title { margin-bottom: 24px !important; }
"""


def create_interface():
    with gr.Blocks(
        title="Llama Stack Chat Interface",
        theme=gr.themes.Base(primary_hue="green").set(
            body_background_fill="#ffffff",
            background_fill_primary="#ffffff",
            background_fill_secondary="#f7f7f8",
            border_color_primary="#e5e5e5",
            block_label_text_color="#6e6e80",
            block_title_text_color="#000000",
            button_primary_background_fill="#10a37f",
            button_primary_background_fill_hover="#0e8e6d",
            button_secondary_background_fill="#f2f2f2",
            button_secondary_background_fill_hover="#e5e5e5",
            button_secondary_text_color="#000000",
        ),
        css=CUSTOM_CSS,
    ) as interface:

        # Header
        gr.HTML(
            """
        <div class="header-title">
            <div style="font-size: 32px; font-weight: 600; color: #000000; margin-bottom: 8px;">
                Llama Stack Chat
            </div>
            <div style="font-size: 16px; color: #6e6e80; font-weight: normal;">
                Powered by Haystack Integration
            </div>
        </div>
        """
        )

        # Configuration inputs
        with gr.Row():
            url = gr.Textbox(
                label="Llama Stack Server URL",
                value="http://0.0.0.0:8321/v1/openai",
                container=True,
                scale=3,
            )
            model = gr.Textbox(
                label="Model",
                value="meta-llama/Llama-3.2-3B-Instruct",
                container=True,
                scale=3,
            )
            connect_btn = gr.Button("Initialize", variant="primary", scale=1)

        # Status and chat interface
        status = gr.HTML('<div class="status-indicator">⏳ Ready to connect...</div>')

        chatbot = gr.Chatbot(
            height=550,
            show_copy_button=True,
            avatar_images=[None, None],
            container=True,
            type="messages",
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Message Llama Stack...",
                show_label=False,
                scale=4,
                container=False,
                min_width=0,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear", variant="secondary", scale=1)

        # Event handlers
        def handle_connect(url, model):
            success, message = chat_interface.initialize_client(url, model)
            status_class = "status-success" if success else "error-message"
            icon = "✅" if success else "❌"
            return (
                f'<div class="status-indicator {status_class}">{icon} {message}</div>'
            )

        def handle_submit(message, history, url, model):
            if message and message.strip():
                for updated_history in chat_interface.chat(
                    message, history, url, model
                ):
                    yield updated_history, ""
            else:
                yield history, ""

        def handle_clear():
            return chat_interface.clear_chat(), ""

        # Bind events
        connect_btn.click(handle_connect, [url, model], [status])
        send_btn.click(handle_submit, [msg, chatbot, url, model], [chatbot, msg])
        msg.submit(handle_submit, [msg, chatbot, url, model], [chatbot, msg])
        clear_btn.click(handle_clear, outputs=[chatbot, msg])

        # Instructions section
        gr.HTML(
            """
        <div style="background-color: #f7f7f8; border: 1px solid #e5e5e5; border-radius: 8px;
                    padding: 20px; margin-top: 24px; color: #000000;">
            <h3 style="color: #000000; margin-bottom: 15px; font-weight: 600;">Instructions</h3>
            <ol style="margin-left: 20px;">
                <li style="margin-bottom: 12px; line-height: 1.5;">
                    <strong>Initialize:</strong> Enter the Llama Stack Server URL and model name, then click "Initialize".
                </li>
                <li style="margin-bottom: 12px; line-height: 1.5;">
                    <strong>Chat:</strong> Type your message and press Enter or click "Send".
                </li>
                <li style="margin-bottom: 12px; line-height: 1.5;">
                    <strong>Clear:</strong> Use "Clear" to reset the conversation.
                </li>
            </ol>
            <p style="margin-top: 15px; color: #6e6e80;">
                <strong>Note:</strong> Make sure your Llama Stack server is running at the specified URL.
            </p>
        </div>
        """
        )

    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860, share=True, debug=True)
