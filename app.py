
import os
import re
import streamlit as st

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API key not found.")
    st.info("Please create a .env file and add: GOOGLE_API_KEY=your_api_key")
    st.stop()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Content Creator",
    page_icon="✨",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #dddddd;
        background-color: #fafafa;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">✨ AI Content Creator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Create simple, creative and engaging content with AI</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Content Settings")

    content_type = st.selectbox(
        "Choose Content Type",
        [
            "Kids Video Script",
            "Story",
            "Educational Content",
            "YouTube Script",
            "Short Story",
            "Poem",
            "Fun Rhyme"
        ]
    )

    target_age = st.selectbox(
        "Target Age",
        [
            "3–5 years",
            "5–7 years",
            "7–10 years",
            "10–13 years"
        ]
    )

    tone = st.selectbox(
        "Tone",
        [
            "Fun and Energetic",
            "Funny",
            "Educational",
            "Friendly",
            "Storytelling",
            "Creative"
        ]
    )

    length = st.selectbox(
        "Content Length",
        [
            "Short",
            "Medium",
            "Long"
        ]
    )

    st.divider()

    st.info(
        "The AI is instructed to use simple English and return only clean content."
    )


# ============================================================
# USER INPUT
# ============================================================

st.subheader("📝 Enter Your Topic")

topic = st.text_area(
    "What content do you want to create?",
    placeholder="Example: Silly rhymes about animals",
    height=120
)


# ============================================================
# GRAPH STATE
# ============================================================

class ContentState(TypedDict, total=False):

    topic: str
    content_type: str
    target_age: str
    tone: str
    length: str

    strategy: str
    generated_content: str
    final_content: str


# ============================================================
# GEMINI MODEL
# ============================================================

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.8
)


# ============================================================
# CLEAN AI RESPONSE
# ============================================================

def clean_ai_response(response):
    """
    Extract only the useful text from the Gemini response.

    This prevents Streamlit from displaying:
    {'type': 'text', 'text': ..., 'extras': ...}
    """

    try:

        # LangChain AIMessage normally has .content
        if hasattr(response, "content"):
            content = response.content
        else:
            content = response

        # Sometimes content is a list
        if isinstance(content, list):

            text_parts = []

            for item in content:

                if isinstance(item, dict):

                    if item.get("type") == "text":
                        text_parts.append(
                            str(item.get("text", ""))
                        )

                elif isinstance(item, str):

                    text_parts.append(item)

            content = "\n".join(text_parts)

        # Make sure the result is a string
        if not isinstance(content, str):
            content = str(content)

        content = content.strip()

        return content

    except Exception:

        return str(response).strip()


# ============================================================
# REMOVE CODE BLOCKS
# ============================================================

def remove_code_blocks(text):
    """
    Removes accidental Markdown code fences.
    """

    text = re.sub(
        r"```[a-zA-Z0-9_+-]*",
        "",
        text
    )

    text = text.replace("```", "")

    return text.strip()


# ============================================================
# CONTENT STRATEGY AGENT
# ============================================================

def create_strategy(state: ContentState):

    topic = state["topic"]
    content_type = state["content_type"]
    target_age = state["target_age"]
    tone = state["tone"]
    length = state["length"]

    strategy_prompt = f"""
You are an expert children's content strategist.

Create a simple content plan for the following:

Topic: {topic}
Content Type: {content_type}
Target Age: {target_age}
Tone: {tone}
Length: {length}

The final content will be shown to children and parents.

Create a short and useful strategy covering:

1. Main idea
2. Opening hook
3. Main learning goal
4. Important sections
5. Interactive element
6. Ending idea

IMPORTANT:

Use ONLY simple English.

Use short sentences.

Do not use difficult words.

Do not use another language.

Do not write Python code.

Do not write JSON.

Do not write dictionaries.

Do not include API information.

Do not include signatures.

Do not include metadata.

Keep the strategy practical for the content writer.
"""

    response = model.invoke(strategy_prompt)

    strategy = clean_ai_response(response)

    return {
        "strategy": strategy
    }


# ============================================================
# CONTENT GENERATOR AGENT
# ============================================================

def generate_content(state: ContentState):

    topic = state["topic"]
    content_type = state["content_type"]
    target_age = state["target_age"]
    tone = state["tone"]
    length = state["length"]
    strategy = state["strategy"]

    generator_prompt = f"""
You are a professional children's content writer.

Create the final content using the information below.

TOPIC:
{topic}

CONTENT TYPE:
{content_type}

TARGET AGE:
{target_age}

TONE:
{tone}

LENGTH:
{length}

CONTENT STRATEGY:
{strategy}

==================================================
VERY IMPORTANT WRITING RULES
==================================================

Write ONLY in simple, natural English.

The content must be easy for a young child to understand.

Use short sentences.

Use common everyday words.

Avoid difficult vocabulary.

Avoid long and complicated sentences.

Make the content sound like a real human wrote it.

Make it fun, warm, creative and engaging.

Use age-appropriate humor.

Use clear examples.

Keep the child interested from beginning to end.

If the content is educational, teach the idea in a very simple way.

If the content is a story, give it a clear beginning, middle and ending.

If the content is a poem or rhyme, make the rhymes natural and easy to say.

If the content is a YouTube script, include useful visual and sound suggestions only when needed.

If asking children questions, give them time to think before revealing the answer.

==================================================
STRICT OUTPUT RULES
==================================================

RETURN ONLY THE FINAL CONTENT.

DO NOT return Python code.

DO NOT return JSON.

DO NOT return a Python dictionary.

DO NOT return XML.

DO NOT return API responses.

DO NOT return metadata.

DO NOT return "type".

DO NOT return "text".

DO NOT return "extras".

DO NOT return "signature".

DO NOT return internal reasoning.

DO NOT explain how you created the content.

DO NOT mention that you are an AI.

DO NOT include technical information.

DO NOT use Hindi.

DO NOT use Telugu.

DO NOT use Tamil.

DO NOT use any other language.

USE ONLY SIMPLE ENGLISH.

Do not put the answer inside a code block.

The output must be ready to show directly to the user.
"""

    response = model.invoke(generator_prompt)

    generated_content = clean_ai_response(response)

    generated_content = remove_code_blocks(
        generated_content
    )

    return {
        "generated_content": generated_content
    }


# ============================================================
# QUALITY CHECKER AGENT
# ============================================================

def check_content(state: ContentState):

    content = state["generated_content"]

    checker_prompt = f"""
You are a professional children's content editor.

Review the content below.

CONTENT:
{content}

Your job is to improve the content if necessary.

Check for:

1. Simple English
2. Short sentences
3. Easy vocabulary
4. Clear meaning
5. Child-friendly language
6. Good grammar
7. Natural flow
8. Fun and engaging writing
9. Age-appropriate humor
10. No confusing words
11. No unnecessary repetition
12. No other languages

IMPORTANT:

Rewrite the content if needed.

Make it sound natural.

Keep the original topic.

Keep the content type.

Do not make it too complicated.

DO NOT add technical explanations.

DO NOT add Python code.

DO NOT add JSON.

DO NOT add dictionaries.

DO NOT add metadata.

DO NOT add signatures.

DO NOT add API information.

DO NOT mention this editing process.

RETURN ONLY THE FINAL CLEAN CONTENT.

USE ONLY SIMPLE ENGLISH.

Do not use any other language.
"""

    response = model.invoke(checker_prompt)

    final_content = clean_ai_response(response)

    final_content = remove_code_blocks(
        final_content
    )

    return {
        "final_content": final_content
    }


# ============================================================
# CREATE LANGGRAPH
# ============================================================

graph_builder = StateGraph(ContentState)

graph_builder.add_node(
    "strategy",
    create_strategy
)

graph_builder.add_node(
    "generator",
    generate_content
)

graph_builder.add_node(
    "checker",
    check_content
)

graph_builder.add_edge(
    START,
    "strategy"
)

graph_builder.add_edge(
    "strategy",
    "generator"
)

graph_builder.add_edge(
    "generator",
    "checker"
)

graph_builder.add_edge(
    "checker",
    END
)

graph = graph_builder.compile()


# ============================================================
# GENERATE BUTTON
# ============================================================

st.divider()

generate_button = st.button(
    "✨ Generate Content",
    type="primary",
    use_container_width=True
)


# ============================================================
# GENERATE CONTENT
# ============================================================

if generate_button:

    if not topic.strip():

        st.warning(
            "⚠️ Please enter a topic first."
        )

    else:

        initial_state: ContentState = {

            "topic": topic.strip(),

            "content_type": content_type,

            "target_age": target_age,

            "tone": tone,

            "length": length
        }

        with st.spinner(
            "✨ Creating your content..."
        ):

            try:

                result = graph.invoke(
                    initial_state
                )

                final_content = result.get(
                    "final_content",
                    ""
                )

                if not final_content:

                    st.error(
                        "The AI did not return any content. Please try again."
                    )

                else:

                    st.session_state[
                        "final_content"
                    ] = final_content

                    st.success(
                        "🎉 Content generated successfully!"
                    )

            except Exception as e:

                st.error(
                    "Something went wrong while generating the content."
                )

                st.exception(e)


# ============================================================
# DISPLAY FINAL CONTENT
# ============================================================

if "final_content" in st.session_state:

    st.divider()

    st.subheader(
        "🎬 Your Generated Content"
    )

    st.markdown(
        '<div class="result-box">',
        unsafe_allow_html=True
    )

    st.markdown(
        st.session_state["final_content"]
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    st.divider()

    # ========================================================
    # COPYABLE TEXT AREA
    # ========================================================

    st.subheader(
        "📋 Copy Your Content"
    )

    st.text_area(
        "Generated Content",
        value=st.session_state["final_content"],
        height=400
    )

    # ========================================================
    # CLEAR BUTTON
    # ========================================================

    if st.button(
        "🗑️ Clear Content",
        use_container_width=True
    ):

        del st.session_state[
            "final_content"
        ]

        st.rerun()
````
