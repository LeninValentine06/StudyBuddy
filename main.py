import tkinter as tk
from tkinter import scrolledtext, messagebox, font
import threading
from google import genai
import time

# Initialize the client with API key
client = genai.Client(api_key="AIzaSyCcXz01pVbNOigauDYzanWePYU9a8ot9B0")


# Create a chat session
chat = client.chats.create(model="gemini-2.0-flash")



def send_message():
    user_question = entry.get()
    if user_question.strip() == '':
        return
    
    # Disable send button while processing
    send_button.config(state=tk.DISABLED)
    
    # Clear the entry field
    entry.delete(0, tk.END)
    
    # Show user message in chat
    display_message("You: " + user_question, user_bg="#90CAF9", align="right")
    
    # Show thinking indicator
    status_label.config(text="🤔 Thinking...", fg="#FF6B6B")
    
    # Process in a separate thread to keep UI responsive
    threading.Thread(target=get_ai_response, args=(user_question,), daemon=True).start()

def get_ai_response(user_question):
    try:
        # Send message to AI
        response = chat.send_message_stream(user_question)
        
        # Collect the response
        response_text = ""
        for chunk in response:
            response_text += chunk.text + " "
        
        # Display the AI response in the main thread
        window.after(0, lambda: display_message("Buddy: " + response_text, ai_bg="#A5D6A7", align="left"))
    except Exception as e:
        # Show error message
        window.after(0, lambda: messagebox.showerror("Oops!", "Something went wrong. Please try again!"))
    finally:
        # Re-enable the send button
        window.after(0, lambda: send_button.config(state=tk.NORMAL))
        window.after(0, lambda: status_label.config(text="Ready to chat!", fg="#4CAF50"))

def display_message(message, user_bg=None, ai_bg=None, align="left"):
    # Configure tag for text alignment
    if align == "right":
        chat_box.tag_configure("right", justify="right")
        tag = "right"
        bg_color = user_bg
    else:
        chat_box.tag_configure("left", justify="left")
        tag = "left"
        bg_color = ai_bg
    
    # Insert message with background color
    chat_box.config(state=tk.NORMAL)
    
    # Add some space before message
    chat_box.insert(tk.END, "\n")
    
    # Insert the message with the appropriate tag and background
    start_pos = chat_box.index(tk.END)
    chat_box.insert(tk.END, message + "\n\n", tag)
    end_pos = chat_box.index(tk.END)
    
    # Add background color to the message
    chat_box.tag_add("bg_color", start_pos, end_pos)
    chat_box.tag_configure("bg_color", background=bg_color)
    
    chat_box.config(state=tk.DISABLED)
    chat_box.see(tk.END)  # Scroll to the end

def on_entry_click(event):
    if entry.get() == "Type your question here...":
        entry.delete(0, tk.END)
        entry.config(fg="black")

def on_focus_out(event):
    if entry.get() == "":
        entry.insert(0, "Type your question here...")
        entry.config(fg="gray")

def on_enter(event):
    send_message()

# Create the main window
window = tk.Tk()
window.title("Friendly Learning Buddy")
window.geometry("700x800")
window.configure(bg="#FFECB3")  # Warm, friendly background

# Try to set Comic Sans MS as the default font, with fallbacks
try:
    default_font = font.nametofont("TkDefaultFont")
    default_font.configure(family="Comic Sans MS", size=12)
except:
    pass  # If Comic Sans MS is not available, stick with system default

# Header
header_frame = tk.Frame(window, bg="#FFD54F", height=80)
header_frame.pack(fill=tk.X)

title_label = tk.Label(header_frame, text="My Learning Buddy", 
                      font=("Comic Sans MS", 24, "bold"), bg="#FFD54F", fg="#5D4037")
title_label.pack(pady=15)

# Create a scrolled text widget for chat
chat_frame = tk.Frame(window, bg="#FFF8E1")
chat_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

chat_box = scrolledtext.ScrolledText(
    chat_frame, 
    wrap=tk.WORD,
    font=("Comic Sans MS", 12),
    bg="#FFF8E1",
    width=40,
    height=20
)
chat_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
chat_box.config(state=tk.DISABLED)

# Status frame
status_frame = tk.Frame(window, bg="#FFECB3")
status_frame.pack(fill=tk.X, padx=20, pady=5)

# Status label 
status_label = tk.Label(status_frame, text="Ready to chat!", font=("Comic Sans MS", 10), bg="#FFECB3", fg="#4CAF50")
status_label.pack(pady=2)

# Create an input frame
input_frame = tk.Frame(window, bg="#FFD54F", height=100)
input_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)

# Input field with prompt
entry = tk.Entry(input_frame, font=("Comic Sans MS", 14), bd=2, relief=tk.GROOVE, width=40)
entry.insert(0, "Type your question here...")
entry.pack(side=tk.LEFT, padx=20, pady=20)
entry.bind("<FocusIn>", on_entry_click)
entry.bind("<FocusOut>", on_focus_out)
entry.bind("<Return>", on_enter)
entry.config(fg="gray")  # Set initial text color to gray

# Send button with a fun look
send_button = tk.Button(input_frame, text="Ask!", command=send_message,
                       font=("Comic Sans MS", 14, "bold"),
                       bg="#FF9800", fg="white", 
                       activebackground="#F57C00",
                       relief=tk.RAISED, bd=3,
                       padx=15, pady=5)
send_button.pack(side=tk.LEFT, padx=10, pady=20)

# Add welcome message
display_message("Buddy: Hi there! I'm your Learning Buddy! 🤖 Ask me anything and I'll help you learn. What would you like to know about today?", ai_bg="#A5D6A7", align="left")

# Start the application
window.mainloop()