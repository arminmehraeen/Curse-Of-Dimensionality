# Curse of Dimensionality Explorer

An interactive web application that demonstrates and visualizes the curse of dimensionality phenomenon using modern data visualization techniques.

## 🌟 Features

- Interactive web interface built with Streamlit
- Real-time visualization of distance metrics in high-dimensional spaces
- Multiple visualization types:
  - Distance ratio analysis
  - Distance distribution metrics
  - K-Nearest Neighbor analysis
- Adjustable parameters:
  - Number of points
  - Maximum dimensions
  - Dimension step size
  - K-Nearest Neighbors

## 📋 Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/arminmehraeen/Curse-Of-Dimensionality.git
cd Curse-Of-Dimensionality
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the sliders in the sidebar to adjust the parameters and observe the changes in real-time

## 📊 Understanding the Visualizations

1. **Distance Ratio Plot**: Shows how the ratio between maximum and minimum distances changes with dimensionality
2. **Distance Distribution Plot**: Displays minimum, maximum, and mean distances across different dimensions
3. **K-NN Distance Plot**: Illustrates how nearest neighbor distances behave in high-dimensional spaces

## 🔍 The Curse of Dimensionality

The curse of dimensionality refers to various phenomena that arise when analyzing and organizing data in high-dimensional spaces. This application demonstrates several key aspects:

- As dimensions increase, the ratio between maximum and minimum distances approaches 1
- The mean distance between points increases with dimensionality
- K-nearest neighbor distances become less meaningful in higher dimensions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Armin Mehraeen - Initial work

## 🙏 Acknowledgments

- Streamlit for the web framework
- Plotly for interactive visualizations
- scikit-learn for nearest neighbor calculations
