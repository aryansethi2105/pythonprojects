import streamlit as st
import random

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
symbols = ['!', '#', '$', '%', '&', '(', ')', '*', '+']

st.title("Password Generator", text_alignment = "center")

# Initialize session state for form values
if "nr_letters" not in st.session_state:
    st.session_state.nr_letters = 1
if "nr_numbers" not in st.session_state:
    st.session_state.nr_numbers = 1
if "nr_symbols" not in st.session_state:
    st.session_state.nr_symbols = 1
if "password_generated" not in st.session_state:
    st.session_state.password_generated = ""

def clear_input():
    st.session_state.nr_letters = 1
    st.session_state.nr_numbers = 1
    st.session_state.nr_symbols = 1
    st.session_state.password_generated = ""

with st.form(key="password_generator"):
    nr_letters = st.number_input(label = "**Enter the number of letters you would like to include in your password:**", min_value=1, key = "nr_letters")
    nr_numbers = st.number_input(label = "**Enter the number of numbers you would like to include in your password:**", min_value = 1, key = "nr_numbers")
    nr_symbols = st.number_input(label = "**Enter the number of symbols you would like to include in your password:**", min_value = 1, key = "nr_symbols")

    col1, col2 = st.columns(2)
    with col1:
        submit = st.form_submit_button(label = "Generate Password", type = "primary")
    with col2:
        clear = st.form_submit_button(label = "Clear", on_click = clear_input)

    # Generate password only when submit is clicked
    if submit:
        password_list = []
        
        for character in range(0, nr_letters):
            password_list.append(random.choice(letters))
        
        for character in range(0, nr_symbols):
            password_list.append(random.choice(symbols))
        
        for character in range(0, nr_numbers):
            password_list.append(random.choice(numbers))
        
        random.shuffle(password_list)
        
        password_hard = ""
        for character in password_list:
            password_hard += character
        
        st.session_state.password_generated = password_hard

# Display password
if st.session_state.password_generated:
    st.write("**Your Generated Password is:**")
    st.code(st.session_state.password_generated, language = None)
