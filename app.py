import os
import streamlit as st

from typing import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Content Creator Agent",
    page_icon="🎬",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🎬 AI Content Creator Agent")

st.write(
    "Create platform-specific content using a multi-agent "
    "AI workflow powered by LangChain and LangGraph."
)

st.divider()


# =========================================================
# API KEY
# =========================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.warning(
        "Gemini API key is not configured. "
        "Please add GEMINI_API_KEY to your environment variables."
    )
    st.stop()


# =========================================================
# GEMINI MODEL
# =========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    temperature=0.7
)


# =========================================================
# LANGGRAPH STATE
# =========================================================

class ContentState(TypedDict):

    topic: str
    platform: str
    audience: str
    tone: str
    content_type: str
    goal: str

    strategy: str
    draft: str
    feedback: str
    final_content: str


# =========================================================
# PLANNER AGENT
# =========================================================

def planner_agent(state: ContentState):

    prompt = f"""
You are the Content Planning Agent.

Analyze the following content requirements:

Topic: {state['topic']}
Platform: {state['platform']}
Target Audience: {state['audience']}
Tone: {state['tone']}
Content Type: {state['content_type']}
Goal: {state['goal']}

Create a content strategy.

Your strategy should include:

1. Target audience analysis
2. Suitable content approach
3. Strong hook idea
4. Key points to cover
5. Recommended call-to-action
6. Platform-specific strategy

Do not write the complete content yet.
Only create the strategy.
"""

    response = llm.invoke(prompt)

    return {
        "strategy": response.content
    }


# =========================================================
# CONTENT CREATOR AGENT
# =========================================================

def creator_agent(state: ContentState):

    prompt = f"""
You are the Content Creation Agent.

Create the actual content using the following information.

Topic:
{state['topic']}

Platform:
{state['platform']}

Target Audience:
{state['audience']}

Tone:
{state['tone']}

Content Type:
{state['content_type']}

Goal:
{state['goal']}

CONTENT STRATEGY:

{state['strategy']}

Create high-quality, engaging content.

Make sure the content:

- Matches the selected platform
- Matches the target audience
- Uses the requested tone
- Has a strong opening
- Is easy to understand
- Includes an appropriate call-to-action
"""

    response = llm.invoke(prompt)

    return {
        "draft": response.content
    }


# =========================================================
# CRITIC AGENT
# =========================================================

def critic_agent(state: ContentState):

    prompt = f"""
You are a Content Critic Agent.

Review the following content.

CONTENT:

{state['draft']}

Evaluate it based on:

1. Hook strength
2. Audience relevance
3. Platform suitability
4. Clarity
5. Engagement potential
6. Call-to-action
7. Overall quality

Identify weaknesses and provide specific improvement suggestions.

Do not rewrite the content.
Only provide constructive feedback.
"""

    response = llm.invoke(prompt)

    return {
        "feedback": response.content
    }


# =========================================================
# OPTIMIZER AGENT
# =========================================================

def optimizer_agent(state: ContentState):

    prompt = f"""
You are the Final Content Optimization Agent.

Improve the content using the critic's feedback.

ORIGINAL CONTENT:

{state['draft']}

CRITIC FEEDBACK:

{state['feedback']}

Requirements:

- Keep the original topic
- Improve the hook
- Improve clarity
- Make it more engaging
- Match the target platform
- Match the target audience
- Keep the requested tone
- Add a strong call-to-action
- Remove unnecessary content

Return ONLY the final improved content.
"""

    response = llm.invoke(prompt)

    return {
        "final_content": response.content
    }


# =========================================================
# CREATE LANGGRAPH WORKFLOW
# =========================================================

def create_workflow():

    workflow = StateGraph(ContentState)

    # Add agents as nodes

    workflow.add_node(
        "planner",
        planner_agent
    )

    workflow.add_node(
        "creator",
        creator_agent
    )

    workflow.add_node(
        "critic",
        critic_agent
    )

    workflow.add_node(
        "optimizer",
        optimizer_agent
    )

    # Workflow connections

    workflow.add_edge(
        START,
        "planner"
    )

    workflow.add_edge(
        "planner",
        "creator"
    )

    workflow.add_edge(
        "creator",
        "critic"
    )

    workflow.add_edge(
        "critic",
        "optimizer"
    )

    workflow.add_edge(
        "optimizer",
        END
    )

    return workflow.compile()


# =========================================================
# STREAMLIT USER INTERFACE
# =========================================================

st.header("📝 Create Your Content")


col1, col2 = st.columns(2)


with col1:

    topic = st.text_input(
        "📝 Topic",
        placeholder="Example: AI for college students"
    )

    platform = st.selectbox(
        "📱 Platform",
        [
            "Instagram",
            "YouTube",
            "LinkedIn",
            "X (Twitter)",
            "Blog"
        ]
    )

    content_type = st.selectbox(
        "📌 Content Type",
        [
            "Social Media Post",
            "Caption",
            "Reel Script",
            "YouTube Script",
            "Carousel",
            "Blog Article"
        ]
    )


with col2:

    audience = st.text_input(
        "🎯 Target Audience",
        placeholder="Example: College students"
    )

    tone = st.selectbox(
        "🎨 Tone",
        [
            "Professional",
            "Friendly",
            "Funny",
            "Educational",
            "Motivational",
            "Creative"
        ]
    )

    goal = st.selectbox(
        "🚀 Content Goal",
        [
            "Increase Engagement",
            "Educate Audience",
            "Build Personal Brand",
            "Increase Followers",
            "Promote a Product",
            "Generate Leads"
        ]
    )


st.divider()


# =========================================================
# GENERATE BUTTON
# =========================================================

if st.button(
    "🚀 Generate Content",
    use_container_width=True
):

    if not topic:

        st.error("Please enter a topic.")

        st.stop()


    if not audience:

        st.error("Please enter your target audience.")

        st.stop()


    initial_state = {

        "topic": topic,

        "platform": platform,

        "audience": audience,

        "tone": tone,

        "content_type": content_type,

        "goal": goal,

        "strategy": "",

        "draft": "",

        "feedback": "",

        "final_content": ""
    }


    # =====================================================
    # RUN LANGGRAPH
    # =====================================================

    with st.spinner(
        "🤖 AI agents are working on your content..."
    ):

        try:

            graph = create_workflow()

            result = graph.invoke(
                initial_state
            )


            # =================================================
            # DISPLAY RESULTS
            # =================================================

            st.success(
                "✅ Content successfully created!"
            )


            st.subheader(
                "🧠 Content Strategy"
            )

            st.write(
                result["strategy"]
            )


            st.subheader(
                "✍️ Initial Draft"
            )

            st.write(
                result["draft"]
            )


            st.subheader(
                "🔍 Critic Feedback"
            )

            st.write(
                result["feedback"]
            )


            st.subheader(
                "🎯 Final Optimized Content"
            )

            st.markdown(
                result["final_content"]
            )


        except Exception as e:

            st.error(
                "Something went wrong while running the AI agents."
            )

            st.code(
                str(e)
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🤖 AI Content Creator Agent | "
    "LangChain + LangGraph + Google Gemini + Streamlit"
)
