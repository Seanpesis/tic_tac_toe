# Hand Gesture Controlled Tic-Tac-Toe

![Project Logo](path_to_your_logo_or_screenshot.png)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)
- [Challenges](#challenges)
- [Author](#author)
- [License](#license)

## Introduction

I developed a **Hand Gesture Controlled Tic-Tac-Toe** game using Python, OpenCV, and Mediapipe. This project allows users to play Tic-Tac-Toe by using hand gestures, enhancing the interactivity and providing a unique gaming experience.

Starting from scratch, I delved into computer vision and machine learning, leveraging numerous tutorials and videos to guide me through the process. This project not only enhanced my programming skills but also provided hands-on experience with real-time gesture recognition and interactive game design.

## Features

- **Real-Time Hand Gesture Detection:** Utilizes Mediapipe for accurate and swift hand gesture recognition.
- **Interactive GUI:** Built with OpenCV, providing a seamless and responsive user interface.
- **Dynamic Game Logic:** Enables users to compete against the computer with intelligent move selections.
- **Symbol Selection:** Allows users to choose their preferred symbol (X or O) before starting the game.
- **Responsive Feedback:** Provides visual cues and instructions to guide users during gameplay.

## Technologies Used

- **Python 3.10**
- **OpenCV:** For image and video processing.
- **Mediapipe:** For real-time hand gesture recognition.
- **NumPy:** For numerical operations.

## Installation

### Prerequisites

- **Python 3.7 to 3.11 (64-bit):** Mediapipe supports these Python versions. You can download Python from the [official website](https://www.python.org/downloads/).
- **Git:** To clone the repository. Download from [here](https://git-scm.com/downloads).

### Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/hand-gesture-tic-tac-toe.git
Navigate to the Project Directory:

bash
Copy code
cd hand-gesture-tic-tac-toe
Create a Virtual Environment:

bash
Copy code
python -m venv myenv
Activate the Virtual Environment:

Windows:

bash
Copy code
myenv\Scripts\activate
macOS/Linux:

bash
Copy code
source myenv/bin/activate
Upgrade pip:

bash
Copy code
python -m pip install --upgrade pip
Install Required Packages:

bash
Copy code
pip install -r requirements.txt
If requirements.txt is not provided, install the packages manually:

bash
Copy code
pip install opencv-python mediapipe numpy
Usage
Activate the Virtual Environment:

Ensure you're in the project directory and the virtual environment is activated.

Windows:

bash
Copy code
myenv\Scripts\activate
macOS/Linux:

bash
Copy code
source myenv/bin/activate
Run the Game:

bash
Copy code
python tic_tac_toe.py
Gameplay Instructions:

Start Screen: Press 'S' to start the game.
Choose Symbol: Press 'O' for Circle or 'X' for Cross.
Place Symbol: Point your finger inside a cell for 1 second to place your chosen symbol.
Winning the Game: The game will declare the winner and provide options to restart or quit.
Demo

You can also include a GIF or link to a video demonstration here.

Challenges
Hand Gesture Recognition: Ensuring accurate and swift recognition of gestures in real-time was challenging. Mediapipe's robust algorithms were instrumental in overcoming this.
GUI Development: Creating an intuitive and responsive interface using OpenCV required a good understanding of image processing techniques.
Compatibility Issues: Encountered issues with Python version compatibility, specifically with Mediapipe not supporting Python 3.13. Switching to Python 3.10 resolved this.
Author
Shon Pasis

LinkedIn
GitHub
Email
License
This project is licensed under the MIT License.

Feel free to customize the sections, add more details, screenshots, or any other relevant information to make your repository more informative and appealing. If you have a live demo or a video walkthrough, including links to those can also enhance the attractiveness of your project.











