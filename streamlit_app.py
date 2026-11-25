import asyncio
import os
import time
from pathlib import Path
import json
import inngest
import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def load_css():
    css_path = Path(__file__).parent / "styles.css"
    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css()


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_path.write_bytes(file.getbuffer())
    return file_path


async def send_event(name: str, data: dict):
    client = inngest.Inngest(
        app_id="rag_app",
        is_production=False
    )

    result = await client.send(
        inngest.Event(
            name=name,
            data=data
        )
    )

    return result[0] if result else None


def inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    response = requests.get(f"{inngest_api_base()}/events/{event_id}/runs")
    response.raise_for_status()
    return response.json().get("data", [])


def wait_for_run_output(
    event_id: str,
    timeout_s: float = 300.0,
    poll_interval_s: float = 0.5
) -> dict:

    start, last_status = time.time(), None

    while time.time() - start <= timeout_s:

        runs = fetch_runs(event_id)

        if runs:
            run = runs[0]

            status = run.get("status")
            last_status = status or last_status

            if status in {
                "Completed",
                "Succeeded",
                "Success",
                "Finished"
            }:

                output = run.get("output")

                print("[RUN STATUS]", status)
                print("[RUN OUTPUT]", repr(output))

                # Completed, but final output not available yet.
                # Keep polling instead of returning {}.
                if output is None or output == {} or output == "":
                    time.sleep(poll_interval_s)
                    continue

                if isinstance(output, str):
                    return json.loads(output)

                return output

            if status in {"Failed", "Cancelled"}:
                raise RuntimeError(
                    f"Function run {status}"
                )

        time.sleep(poll_interval_s)

    raise TimeoutError(
        f"Timed out waiting for run output "
        f"(last status: {last_status})"
    )


for key, default in {
    "answer": "",
    "sources": [],
    "ingested_files": {},   # file_key -> source_id
}.items():
    st.session_state.setdefault(key, default)

st.markdown(
    """
    <div class="app-header">
        <h1>ContextIQ</h1>
        <p>Intelligent Document Assistant</p>
    </div>
    """,
    unsafe_allow_html=True,
)

upload_column, query_column = st.columns(
    [0.85, 1.15], gap="large", vertical_alignment="top"
)

with upload_column:
    with st.container(border=True):
        st.markdown("### Add a document")
        st.caption("Upload one PDF to add it to your knowledge base.")
        uploaded_files = st.file_uploader("PDF files", type=["pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files:

            for uploaded in uploaded_files:

                file_key = f"{uploaded.name}:{uploaded.size}"

                if file_key in st.session_state.ingested_files:
                    continue

                try:
                    with st.spinner(f"Indexing {uploaded.name}..."):

                        path = save_uploaded_pdf(uploaded)

                        event_id = asyncio.run(
                            send_event(
                                "rag/ingest_pdf",
                                {
                                    "pdf_path": str(path.resolve()),
                                    "source_id": path.name,
                                },
                            )
                        )

                        wait_for_run_output(
                            event_id,
                            timeout_s=300,
                        )

                        st.session_state.ingested_files[file_key] = path.name

                        st.success(f"{path.name} is ready.")

                except Exception as error:
                    st.error(
                        f"Unable to ingest {uploaded.name}: {error}"
                    )

with query_column:
    with st.container(border=True, key="query_panel"):

        st.markdown("### Ask your documents")
        st.caption(
            "Enter a question and choose how many matching chunks should be used."
        )

        with st.form("rag_query_form"):

            question_column, chunks_column = st.columns(
                [4, 1],
                gap="medium",
                vertical_alignment="bottom"
            )

            with question_column:
                question = st.text_input(
                    "Your question",
                    placeholder="Ask something about the uploaded PDF...",
                )

            with chunks_column:
                top_k = st.number_input(
                    "Chunks",
                    1,
                    20,
                    5,
                    1
                )

            submitted = st.form_submit_button("Ask")

        if submitted:

            if not question.strip():
                st.warning("Enter a question before submitting.")

            elif not st.session_state.ingested_files:
                st.warning("Upload at least one PDF before asking a question.")

            else:
                try:
                    with st.spinner(
                        "Finding context and generating an answer..."
                    ):

                        event_id = asyncio.run(
                            send_event(
                                "rag/query_pdf_ai",
                                {
                                    "question": question.strip(),
                                    "top_k": int(top_k),

                                    # NEW
                                    "source_ids": list(
                                        st.session_state.ingested_files.values()
                                    ),
                                },
                            )
                        )

                        output = wait_for_run_output(
                            event_id,
                            timeout_s=300,
                        )
                        #
                        # print("\n[STREAMLIT OUTPUT]")
                        # print(output)
                        # print("[STREAMLIT OUTPUT TYPE]", type(output))

                        answer = output.get("answer", "")

                        if not answer:
                            st.error(
                                f"Inngest completed, but no answer was returned. "
                                f"Output: {output}"
                            )
                        else:
                            st.session_state["answer"] = answer

                            st.session_state["sources"] = list(
                                dict.fromkeys(
                                    output.get("sources", [])
                                )
                            )

                except (requests.RequestException,RuntimeError,TimeoutError) as error:
                    st.error(str(error))

        st.markdown("### Answer")
        if st.session_state.answer:
            st.write(st.session_state.answer)
            if st.session_state.sources:
                st.markdown('<div class="sources-heading">Sources</div>', unsafe_allow_html=True)
                for source in st.session_state.sources:
                    st.write(f"- {source}")
        else:
            st.markdown(
                '<div class="answer-placeholder">Your generated answer will appear here.</div>',
                unsafe_allow_html=True,
            )
