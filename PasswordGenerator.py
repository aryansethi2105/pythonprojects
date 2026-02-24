import streamlit as st
import random

st.title("Password Generator")
st.header("Password Generator", text_alignment = "center")

with st.form(key = "password_generator", clear_on_submit = True):
  nr_letters = st.number_input(label = "Enter the number of letters you would like to include in your password", min_value = 1)
  nr_numbers = st.number_input(label = "Enter the number of numbers you would like to include in your password", min_value = 1)
  nr_ symbols = st.number_input(label = "Enter the number of symbols you would like to include in your password", min_value = 1)
  st.form_submit_button(label = "Generate Password")

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

  st.write(password_hard)
  
