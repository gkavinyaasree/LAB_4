# relu_app.py
# Streamlit app: visualise the ReLU activation function
# Course: BSD3513 Introduction to Artificial Intelligence - Lab Report 5

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ReLU Activation", layout="wide")

# ---------------------------------------------------------
# Activation function and its derivative
# ---------------------------------------------------------
def activation(x):
    """ReLU function."""
    return np.maximum(0, x)


def derivative(x):
    """Derivative of ReLU (used in backpropagation)."""
    return (x > 0).astype(float)


# ---------------------------------------------------------
# Page header
# ---------------------------------------------------------
st.title("ReLU Activation Function")
st.write("ReLU (Rectified Linear Unit) passes positive values through unchanged and maps negative values to zero.")
st.latex(r"f(z) = \max(0, z)")
st.latex(r"f'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \le 0 \end{cases}")

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
    ax.plot(x, y, linewidth=2, label="ReLU(z)")
    if show_derivative:
        ax.plot(x, dy, linestyle="--", linewidth=1.8, label="Derivative")
    if show_point:
        ax.scatter([x_in], [y_in], color="red", zorder=5, label=f"z = {x_in:.1f}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("z (input to activation)")
    ax.set_ylabel("Output")
    ax.set_title("ReLU Activation Function")
    ax.grid(True, alpha=0.4)
    ax.legend()
    st.pyplot(fig)

with col_info:
    st.subheader("Test one neuron")
    st.metric("Input z", f"{x_in:.2f}")
    st.metric("Output ReLU(z)", f"{y_in:.4f}")
    st.metric("Gradient at z", f"{dy_in:.4f}")

    st.subheader("Key properties")
    st.markdown(
        """
- **Output range:** 0 to +infinity
- **Negative inputs:** mapped to 0
- **Positive inputs:** passed through unchanged
- **Gradient:** 1 for z > 0, otherwise 0
"""
    )

# ---------------------------------------------------------
# Explanation
# ---------------------------------------------------------
st.subheader("Why it matters in a neural network")
st.write(
    "ReLU adds non-linearity while being very cheap to compute. Because its gradient is 1 for positive inputs, it avoids the vanishing gradient problem in deep networks, which is why it is the default choice for hidden layers."
)

with st.expander("Advantages and limitations"):
    st.markdown(
        """
**Advantages**
- Very fast to compute
- Helps avoid vanishing gradients for positive inputs
- Produces sparse activations (many neurons output 0)

**Limitations**
- Dying ReLU: a neuron stuck at z <= 0 gets zero gradient and stops learning
- Output is not bounded and not zero-centred
"""
    )
