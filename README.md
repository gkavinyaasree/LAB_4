Three separate Streamlit apps that visualise common neural network activation functions.

| App | File | Output range |
|-----|------|--------------|
| ReLU | `relu_app.py` | 0 to +inf |
| Sigmoid | `sigmoid_app.py` | 0 to 1 |
| Tanh | `tanh_app.py` | -1 to 1 |

## Features
- Interactive plot of the function and its derivative
- Adjustable x-axis range
- Test input slider that shows the neuron output and gradient
- Short explanation of properties, advantages and limitations

## Run locally
```bash
pip install -r requirements.txt
streamlit run relu_app.py
streamlit run sigmoid_app.py
streamlit run tanh_app.py
```

## Live apps
- ReLU: <http://localhost:8503/>
- Sigmoid: <http://localhost:8501/>
- Tanh: <http://localhost:8502/>

