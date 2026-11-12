# tanh_app.py
# Streamlit app: visualise the Tanh activation function
# Course: BSD3513 Introduction to Artificial Intelligence - Lab Report 5

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tanh Activation", layout="wide")

# ---------------------------------------------------------
# Activation function and its derivative
# ---------------------------------------------------------
def activation(x):
    """Tanh function."""
    return np.tanh(x)


def derivative(x):
    """Derivative of Tanh (used in backpropagation)."""
    return 1 - np.tanh(x) ** 2


# ---------------------------------------------------------
# Page header
# ---------------------------------------------------------
st.title("Tanh Activation Function")
st.write("Tanh (hyperbolic tangent) is similar to sigmoid but maps inputs to values between -1 and 1.")
st.latex(r"\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}")
st.latex(r"\tanh'(z) = 1 - \tanh^{2}(z)")

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    x_min, x_max = st.slider("x-axis range", -10.0, 10.0, (-6.0, 6.0), 0.5)
    show_derivative = st.checkbox("Show derivative", value=True)
    show_point = st.checkbox("Show test input on the curve", value=True)

if x_min >= x_max:
    st.error("The x-axis range must have a minimum smaller than its maximum.")
    st.stop()

with st.sidebar:
    x_in = st.slider(
        "Test input z (weighted sum + bias)",
        float(x_min), float(x_max),
        float(np.clip(1.0, x_min, x_max)), 0.1,
    )

# ---------------------------------------------------------
# Compute values
# ---------------------------------------------------------
x = np.linspace(x_min, x_max, 500)
y = activation(x)
dy = derivative(x)

y_in = float(activation(np.array(x_in)))
dy_in = float(derivative(np.array(x_in)))

# ---------------------------------------------------------
# Layout: plot on the left, numbers and notes on the right
# ---------------------------------------------------------
col_plot, col_info = st.columns([2, 1])

with col_plot:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, linewidth=2, label="Tanh(z)")
    if show_derivative:
        ax.plot(x, dy, linestyle="--", linewidth=1.8, label="Derivative")
    if show_point:
        ax.scatter([x_in], [y_in], color="red", zorder=5, label=f"z = {x_in:.1f}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("z (input to activation)")
    ax.set_ylabel("Output")
    ax.set_title("Tanh Activation Function")
    ax.grid(True, alpha=0.4)
    ax.legend()
    st.pyplot(fig)

with col_info:
    st.subheader("Test one neuron")
    st.metric("Input z", f"{x_in:.2f}")
    st.metric("Output Tanh(z)", f"{y_in:.4f}")
    st.metric("Gradient at z", f"{dy_in:.4f}")

    st.subheader("Key properties")
    st.markdown(
        """
- **Output range:** -1 to 1
- **Output at z = 0:** 0 (zero-centred)
- **Shape:** smooth S-curve
- **Maximum gradient:** 1 (at z = 0)
"""
    )

# ---------------------------------------------------------
# Explanation
# ---------------------------------------------------------
st.subheader("Why it matters in a neural network")
st.write(
    "Tanh is zero-centred, so its outputs are balanced around 0. This often makes training easier than with sigmoid in hidden layers, although it still saturates for large |z|."
)

with st.expander("Advantages and limitations"):
    st.markdown(
        """
**Advantages**
- Zero-centred output
- Stronger gradients than sigmoid near z = 0

**Limitations**
- Still suffers from vanishing gradients when |z| is large
- More expensive to compute than ReLU
"""
    )
