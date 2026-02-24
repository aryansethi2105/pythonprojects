import streamlit as st
from st_copy import copy_button
import random

st.title("Password Generator", text_alignment = "center")

with st.form(key = "password_generator", clear_on_submit = True):
  letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
  numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  symbols = ['!', '#', '$', '%', '&', '(', ')', '*', '+']
  nr_letters = st.number_input(label = "Enter the number of letters you would like to include in your password", min_value = 1)
  nr_numbers = st.number_input(label = "Enter the number of numbers you would like to include in your password", min_value = 1)
  nr_symbols = st.number_input(label = "Enter the number of symbols you would like to include in your password", min_value = 1)
  submit = st.form_submit_button(label = "Generate Password")

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
  
if submit:
  st.code(password_hard)
