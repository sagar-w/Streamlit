# installation
## ---->>>   pip install streamlit

# Run file
## ---->>>  streamlit run filename.py

import streamlit as st
import pandas as pd
import numpy as np
import time 

# # Set page title and icon
# st.set_page_config(page_title="My Streamlit App", page_icon=":bar_chart:")

# # Title
# st.title("Streamlit Example App")

# # # Header and subheader
# st.header("This is a header")
# st.subheader("This is a subheader")

# # # Text
# st.write("This is some text.")


# st.subheader('This is a subheader')
# st.subheader('A subheader with _italics_ :blue[colors] and emojis :sunglasses:')


# # # Markdown
# st.markdown("### This is a Markdown title")
# st.markdown("You can use **Markdown** to format your text.")

# # # Data display
# st.subheader("Data Display")

# # # DataFrame
# df = pd.DataFrame({
# 'Name': ['Alice', 'Bob', 'Charlie', 'David'],
#  'Age': [25, 32, 19, 45],
#  'Salary': [50000, 60000, 40000, 80000]
# })

# st.dataframe(df)

# # # Table
# st.subheader("Table Display")
# st.table(df)

# # # Charts
# st.subheader("Charts")

# # # Line chart
# chart_data = pd.DataFrame(
# np.random.randn(20, 3),
#  columns=['A', 'B', 'C']
# )

# st.line_chart(chart_data)

# # # Bar chart
# st.bar_chart(chart_data)

# # # Map
# st.subheader("Map")
# map_data = pd.DataFrame(
#     np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
# columns=['lat', 'lon']
# )
# st.map(map_data)

# # Interactive widgets
# st.subheader("Interactive Widgets")

# # Slider
# slider_value = st.slider("Select a value", 0, 100, 50)

# # Text input
# text_input = st.text_input("Enter text here", "Default text")

# # Selectbox
# options = ['Option 1', 'Option 2', 'Option 3']
# selectbox_value = st.selectbox("Choose an option", options)

# # Checkbox
# checkbox_value = st.checkbox("Check this box")

# # Radio buttons
# radio_value = st.radio("Choose one option", options)

# # Buttons
# if st.button("Click me!"):
#     st.write("You clicked the button!")

# # # Expander
# with st.expander("Click to expand"):
#     st.write("This is hidden content that can be expanded.")

# # # # Progress bar
# # progress_value = st.progress(0)
# # for i in range(101):
# #     progress_value.progress(i)
# #     time.sleep(0.1)

# # # Sidebar
# st.sidebar.title("Sidebar Title")

# # Sidebar widgets
# sidebar_slider = st.sidebar.slider("Sidebar Slider", 0, 100, 50)
# sidebar_text_input = st.sidebar.text_input("Sidebar Text Input", "Default text")
# sidebar_selectbox = st.sidebar.selectbox("Sidebar Selectbox", options)
# sidebar_checkbox = st.sidebar.checkbox("Sidebar Checkbox")
# sidebar_radio = st.sidebar.radio("Sidebar Radio", options)

# # # Placeholders
# st.subheader("Placeholders")
# placeholder = st.empty()
# placeholder.text("This is a placeholder")

# # Show JSON
# st.subheader("Show JSON")
# st.json({"name": "John", "age": 30, "city": "New York"})

# # Show code
# st.subheader("Show Code")
# st.code("""
# def my_function():
#     return "Hello, Streamlit!"

# result = my_function()
# """)


# # Show help
# st.subheader("Show Help")
# st.help(pd.DataFrame)

# # # File upload
# st.subheader("File Upload")
# uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])


# # Info, warning, and error messages
# st.subheader("Messages")
# st.info("This is an informational message.")
# st.warning("This is a warning message.")
# st.error("This is an error message.")

# # Balloons
# st.subheader("Balloons")
# # Buttons
# if st.button("Click to see Balloons!"):
#     st.balloons()

# st.balloons()