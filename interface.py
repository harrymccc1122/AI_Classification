import tkinter as tk
from tkinter import filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from classifier import classify_data

def show_ui():
    window = tk.Tk()
    window.title("Walking / Jumping AI")  # Add a window title
    window.geometry("700x500")  # Set the window size to 800x600 pixels
    window.minsize(700, 500)

    # Prepare main frame)
    button_frame = tk.Frame(window, width=150, background="gray")
    button_frame.pack(side=tk.LEFT, fill=tk.Y)
    main_frame = tk.Frame(window)
    main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    greeting = tk.Label(main_frame, text='Welcome to JumpGPT! Click "Upload File" with a CSV file to tell if you were jumping or walking')
    greeting.pack()

    uploadButton = tk.Button(button_frame, text="Upload File", width=10, command=lambda: upload(main_frame))
    uploadButton.pack(padx = 10, pady = 10)
    SaveAsButton = tk.Button(button_frame, text="Save As", width=10, command=save)
    SaveAsButton.pack(padx = 10, pady = 10)

    window.mainloop()


def save():
    if current_data_file_path == "":
        return

    file_path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[("Comma Separated Value",".csv")])
    classified_data = classify_data("data.h5", current_data_file_path)
    classified_data.to_csv(file_path)


def upload(parent):
    global current_data_file_path
    global canvas_widget
    current_data_file_path = filedialog.askopenfilename(filetypes=[("Comma Separated Value",".csv")])
    classified_data = classify_data("data.h5", current_data_file_path)

    if canvas_widget is not None:
        canvas_widget.destroy()

    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    colors = {
        None: "gray",
        0: "red",
        1: "green",
    }
    
    transition_indices = [i+1 for i in range(0, len(classified_data)-1) if classified_data.iloc[i, -1] != classified_data.iloc[i+1, -1]]
    transition_indices = [0] + transition_indices
    
    print(transition_indices)
    for i in range(0,len(transition_indices)-1):
        category_data = classified_data.iloc[transition_indices[i]:transition_indices[i+1]]
        category = category_data["category"].max()
        ax.plot(
            category_data["Time (s)"],
            category_data["Absolute acceleration (m/s^2)"],
            color=colors[category],
            # label=name[category]
        )

    ax.legend(handles=[
        Line2D([0],[0], color='r', label="Jumping"),
        Line2D([0],[0], color='g', label="Walking"),
    ])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Absolute Acceleration (m/s^2)")

    canvas = FigureCanvasTkAgg(fig, master=parent)  # Embedding in the parent frame
    canvas.draw()
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


def main():
    show_ui()


if __name__ == "__main__":
    current_data_file_path = ""
    canvas_widget = None
    main()