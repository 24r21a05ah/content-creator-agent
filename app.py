import os
import streamlit as st
from google import genai

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Content Creator Agent",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("🎬 AI Content Creator Agent")

st.write(
    "Create engaging content for Instagram, YouTube, LinkedIn, "
    "X, blogs, and other platforms using AI."
)

st.divider()

# ---------------------------------------------------------
# GEMINI API KEY
# ---------------------------------------------------------
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.warning(
        "Gemini API key is not configured. "
        "Please add GEMINI_API_KEY in your environment variables."
    )
    st.stop()

client = genai.Client(api_key=api_key)

# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    topic = st.text_input(
        "📝 Content Topic",
        placeholder="Example: How AI is changing education"
    )

    platform = st.selectbox(
        "📱 Select Platform",
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
            "Post",
            "Caption",
            "Reel Script",
            "YouTube Script",
            "Carousel",
            "Blog Article"
        ]
    )

with col2:

    target_audience = st.text_input(
        "🎯 Target Audience",
        placeholder="Example: College students"
    )

    tone = st.selectbox(
        "🎨 Content Tone",
        [
            "Professional",
            "Friendly",
            "Funny",
            "Educational",
            "Motivational",
            "Creative"
        ]
    )

    language = st.selectbox(
        "🌐 Language",
        [
            "English",
            "Telugu",
            "Hindi"
        ]
    )

# ---------------------------------------------------------
# CONTENT GOAL
# ---------------------------------------------------------

goal = st.selectbox(
    "🚀 Content Goal",
    [
        "Increase Engagement",
        "Educate Audience",
        "Promote a Product",
        "Build Personal Brand",
        "Increase Followers",
        "Generate Leads"
    ]
)

additional_instructions = st.text_area(
    "✨ Additional Instructions",
    placeholder=(
        "Example: Keep it simple, use a strong hook, "
        "include a call-to-action..."
    )
)

st.divider()

# ---------------------------------------------------------
# GENERATE CONTENT
# ---------------------------------------------------------

if st.button(
    "🤖 Generate Content",
    use_container_width=True
):

    if not topic:
        st.error("Please enter a content topic.")
        st.stop()

    if not target_audience:
        st.error("Please enter your target audience.")
        st.stop()

    # -----------------------------------------------------
    # AGENT PROMPT
    # -----------------------------------------------------

    prompt = f"""
You are an intelligent AI Content Creator Agent.

Your task is to create high-quality content based on the
user's requirements.

USER REQUIREMENTS:

Topic:
{topic}

Platform:
{platform}

Content Type:
{content_type}

Target Audience:
{target_audience}

Tone:
{tone}

Language:
{language}

Content Goal:
{goal}

Additional Instructions:
{additional_instructions}

---------------------------------------------------------
AGENTIC CONTENT CREATION PROCESS
---------------------------------------------------------

STEP 1:
Understand the topic and target audience.

STEP 2:
Analyze the selected platform and determine what type of
content works best for that platform.

STEP 3:
Create a strong and attention-grabbing hook.

STEP 4:
Generate the main content.

STEP 5:
Make the content suitable for the target audience.

STEP 6:
Check whether the content matches the selected tone
and platform.

STEP 7:
Improve the content if necessary.

STEP 8:
Add an appropriate call-to-action.

STEP 9:
Suggest relevant hashtags or keywords when appropriate.

---------------------------------------------------------
OUTPUT FORMAT
---------------------------------------------------------

# 🎯 Content Strategy

Explain briefly:
- Target audience
- Platform strategy
- Content goal

# 🪝 Hook

Create a strong opening hook.

# ✍️ Final Content

Generate the complete requested content.

# 📢 Call To Action

Give an appropriate CTA.

# #️⃣ Hashtags / Keywords

Give relevant hashtags or keywords.

# 💡 Improvement Suggestions

Give 2-3 suggestions for improving the content.

IMPORTANT:

- Do not claim that the content will definitely go viral.
- Keep the content natural and engaging.
- Adapt the content to the selected platform.
- Use the requested language.
"""

    # -----------------------------------------------------
    # AI RESPONSE
    # -----------------------------------------------------

    with st.spinner(
        "🤖 AI Content Agent is creating your content..."
    ):

        try:

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            st.success("✅ Your content is ready!")

            st.markdown(response.text)

        except Exception as e:

            st.error(
                "Something went wrong while generating content."
            )

            st.code(str(e))

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "🤖 AI Content Creator Agent | "
    "Built using Streamlit and Google Gemini"
)
