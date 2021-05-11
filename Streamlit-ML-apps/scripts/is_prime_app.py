import streamlit as st

st.title("A primality test app with Streamlit")
"""
## Dr. Tirthajyoti Sarkar, Fremont, CA, July 2020
[My LinkedIn profile](https://www.linkedin.com/in/tirthajyoti-sarkar-2127aa7/),
[My Github profile](https://github.com/tirthajyoti.)

---
"""
st.header("Is it prime?")

st.markdown("We determine primality using a simple code as follows")
st.code('''def is_prime(number):
    """
    Determines primality
    """
    from math import ceil,sqrt
    flag = True
    if number%2==0:
        flag=False
        return flag,0
    sqrt_num = ceil(sqrt(number))
    st.write(f"Checking divisibility up to **{sqrt_num}**")
    for i in range(2,sqrt_num):
        if number%i == 0:
            flag = False
            return flag,i
    return flag,i''',language='python')

number = st.number_input('Insert a number')
number = int(number)
st.write('The current number is ', number)

def is_prime(number):
    """
    Determines primality
    """
    from math import ceil,sqrt
    flag = True
    if number%2==0:
        flag= False
        return flag, 2
    sqrt_num = ceil(sqrt(number))
    st.write(f"Checking divisibility up to **{sqrt_num}**")
    for i in range(2,sqrt_num):
        if number%i == 0:
            flag = False
            return flag,i
    return flag,i

decision, divisor = is_prime(number)

if decision:
    st.markdown("### Yes, the given number is prime")
else:
    st.markdown("### No, the given number is not a prime.")
