import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sigmoid Activation", layout="wide")

# ---------------------------------------------------------
# Activation function and its derivative
# ---------------------------------------------------------
def activation(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))


def derivative(x):
    """Derivative of Sigmoid (used in backpropagation)."""
    return activation(x) * (1 - activation(x))


# ---------------------------------------------------------
# Page header
# ---------------------------------------------------------
st.title("Sigmoid Activation Function")
st.write("Sigmoid squashes any real input into a value between 0 and 1, so the output can be read as a probability.")
st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")
st.latex(r"\sigma'(z) = \sigma(z)\,(1 - \sigma(z))")

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
    ax.plot(x, y, linewidth=2, label="Sigmoid(z)")
    if show_derivative:
        ax.plot(x, dy, linestyle="--", linewidth=1.8, label="Derivative")
    if show_point:
        ax.scatter([x_in], [y_in], color="red", zorder=5, label=f"z = {x_in:.1f}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("z (input to activation)")
    ax.set_ylabel("Output")
    ax.set_title("Sigmoid Activation Function")
    ax.grid(True, alpha=0.4)
    ax.legend()
    st.pyplot(fig)

with col_info:
    st.subheader("Test one neuron")
    st.metric("Input z", f"{x_in:.2f}")
    st.metric("Output Sigmoid(z)", f"{y_in:.4f}")
    st.metric("Gradient at z", f"{dy_in:.4f}")

    st.subheader("Key properties")
    st.markdown(
        """
- **Output range:** 0 to 1
- **Output at z = 0:** 0.5
- **Shape:** smooth S-curve
- **Maximum gradient:** 0.25 (at z = 0)
"""
    )

# ---------------------------------------------------------
# Explanation
# ---------------------------------------------------------
st.subheader("Why it matters in a neural network")
st.write(
    "Sigmoid is used in output layers for binary classification, such as the spam example in the lecture where 0.636 was read as a spam probability. In hidden layers it can cause vanishing gradients, because the gradient becomes tiny when |z| is large."
)

with st.expander("Advantages and limitations"):
    st.markdown(
        """
**Advantages**
- Output between 0 and 1 reads as a probability
- Smooth and differentiable everywhere

**Limitations**
- Vanishing gradients for large positive or negative z
- Output is not zero-centred
- Exponential is slower to compute than ReLU
"""
    )
