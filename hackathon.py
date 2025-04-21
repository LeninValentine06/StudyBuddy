import cv2
import mediapipe as mp
import time
import json
from datetime import datetime
import numpy as np
import os
from ultralytics import YOLO
import threading
import socket
import argparse

class StudyMonitor:
    def __init__(self, session_id=None, tasks=None, schedule=None):
        # Initialize session information
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tasks = tasks or []
        self.schedule = schedule or {}
        
        # Initialize timers and counters
        self.session_start = datetime.now()
        self.current_state = "initializing"
        self.prev_state = None
        self.state_start_time = time.time()
        self.state_times = {"studying": 0, "not_studying": 0, "phone_use": 0}
        
        # Task completion tracking
        self.completed_tasks = []
        
        # Initialize webcam
        print("Initializing webcam...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open webcam")
            
        # Initialize MediaPipe Pose
        print("Setting up MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Load YOLOv8 model
        print("Loading YOLOv8 model...")
        try:
            # Using YOLOv8n (nano) for better performance
            self.yolo_model = YOLO("yolov8n.pt")
            print("YOLOv8 model loaded successfully")
            # Class 67 is 'cell phone' in COCO dataset which YOLOv8 uses
            self.phone_class_id = 67
        except Exception as e:
            print(f"Failed to load YOLOv8 model: {e}")
            print("Will attempt to continue without object detection")
            self.yolo_model = None
        
        # Start socket server for communication with Flutter app
        self.should_exit = False
        self.server_thread = threading.Thread(target=self.start_server)
        self.server_thread.daemon = True
        self.server_thread.start()
    
    def start_server(self):
        """Start a socket server to communicate with the Flutter app"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind(('0.0.0.0', 8080))
            server_socket.listen(5)
            print("Server started on localhost:8080")
            
            while not self.should_exit:
                server_socket.settimeout(1.0)  # Check for exit condition every second
                
                try:
                    client_socket, address = server_socket.accept()
                    print(f"Connection from {address}")
                    
                    # Handle client connection
                    data = client_socket.recv(4096)
                    if data:
                        self.handle_client_data(data)
                    
                    # Send current status back to client
                    status_data = self.get_current_status()
                    client_socket.send(json.dumps(status_data).encode())
                    
                    client_socket.close()
                except socket.timeout:
                    # This is expected due to the timeout, continue the loop
                    continue
                except Exception as e:
                    print(f"Error in socket communication: {e}")
        
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            server_socket.close()
    
    def handle_client_data(self, data):
        """Process data received from the Flutter app"""
        try:
            command = json.loads(data.decode())
            
            if "action" in command:
                if command["action"] == "update_tasks":
                    self.tasks = command.get("tasks", [])
                    print(f"Updated tasks: {self.tasks}")
                
                elif command["action"] == "update_schedule":
                    self.schedule = command.get("schedule", {})
                    print(f"Updated schedule: {self.schedule}")
                
                elif command["action"] == "mark_task_complete":
                    task_id = command.get("task_id")
                    if task_id and task_id not in self.completed_tasks:
                        self.completed_tasks.append(task_id)
                        print(f"Marked task {task_id} as complete")
                
                elif command["action"] == "request_data":
                    # Nothing special needed here, we'll send the current status in the main loop
                    pass
        
        except json.JSONDecodeError:
            print("Received invalid JSON data")
        except Exception as e:
            print(f"Error processing client data: {e}")
    
    def get_current_status(self):
        """Get current monitoring status to send to Flutter app"""
        # Calculate current state time
        current_time = time.time()
        current_state_time = 0
        if self.prev_state:
            current_state_time = current_time - self.state_start_time
        
        # Create state times including current state
        state_times = dict(self.state_times)
        if self.prev_state:
            state_times[self.prev_state] = state_times[self.prev_state] + current_state_time
        
        return {
            "session_id": self.session_id,
            "current_state": self.current_state,
            "session_start": self.session_start.isoformat(),
            "session_duration": round((current_time - self.session_start.timestamp())),
            "state_times": {k: round(v) for k, v in state_times.items()},
            "tasks": self.tasks,
            "completed_tasks": self.completed_tasks,
            "schedule": self.schedule
        }
    
    def update_state_time(self, new_state):
        """Update time spent in current state and switch to new state"""
        current_time = time.time()
        if self.prev_state:
            time_in_state = current_time - self.state_start_time
            self.state_times[self.prev_state] += time_in_state
        
        self.prev_state = new_state
        self.current_state = new_state
        self.state_start_time = current_time
    
    def detect_pose_state(self, landmarks):
        """Determine state based on pose landmarks"""
        if landmarks is None:
            return "not_studying"  # No pose detected
        
        # Extract key landmarks
        try:
            # Get nose, shoulders, and wrists positions
            nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Check if head is down (studying posture)
            head_down = nose.y > (left_shoulder.y + right_shoulder.y) / 2 - 0.1
            
            # Check if hands are near face (more sensitive detection)
            hands_near_face = (
                abs(left_wrist.y - nose.y) < 0.3 and abs(left_wrist.x - nose.x) < 0.3
            ) or (
                abs(right_wrist.y - nose.y) < 0.3 and abs(right_wrist.x - nose.x) < 0.3
            )
            
            # Check if head is turned away (not studying)
            head_turned = abs(nose.x - 0.5) > 0.15
            
            # Return state based on pose analysis
            if head_down and not hands_near_face:
                return "studying"
            elif hands_near_face:
                return "potential_phone_use"  # Will be confirmed with object detection
            else:
                return "not_studying"
                
        except Exception as e:
            print(f"Error detecting pose state: {e}")
            return "not_studying"
    
    def detect_phone(self, frame):
        """Detect if phone is present in the frame using YOLOv8"""
        if self.yolo_model is None:
            return False, frame
        
        # Run inference with YOLOv8
        results = self.yolo_model(frame, conf=0.4)  # Lower confidence threshold
        
        # Check if phone is detected
        phone_detected = False
        
        # Process results
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                # Check if the detected object is a phone (class 67)
                cls = int(box.cls.item())
                conf = box.conf.item()
                
                if cls == self.phone_class_id:
                    phone_detected = True
                    
                    # Draw bounding box if phone is detected
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Phone ({conf:.2f})", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return phone_detected, frame
    
    def determine_state(self, pose_state, phone_detected):
        """Determine final state based on pose and phone detection"""
        if phone_detected:
            # If phone is detected with high confidence, override to phone_use
            return "phone_use"
        elif pose_state == "potential_phone_use":
            # If pose suggests phone use but no phone detected, still mark as phone_use
            # This helps when phone detection occasionally fails
            return "phone_use"
        elif pose_state == "studying":
            return "studying"
        else:
            return "not_studying"
    
    def format_time(self, seconds):
        """Format seconds to MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def display_tasks(self, frame):
        """Display task information on the frame"""
        if not self.tasks:
            return frame
        
        # Create a semi-transparent overlay for tasks
        overlay = frame.copy()
        
        # Draw background rectangle
        cv2.rectangle(overlay, (frame.shape[1] - 350, 10), (frame.shape[1] - 10, 40 + 30 * len(self.tasks)), (0, 0, 0), -1)
        
        # Add title
        cv2.putText(overlay, "TASKS:", (frame.shape[1] - 340, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display each task with completion status
        for i, task in enumerate(self.tasks):
            task_id = task.get("id", str(i))
            task_name = task.get("name", f"Task {i+1}")
            completed = task_id in self.completed_tasks
            
            status_color = (0, 255, 0) if completed else (0, 165, 255)  # Green if completed, orange if not
            status_text = "✓" if completed else "□"
            
            # Draw task text
            cv2.putText(overlay, f"{status_text} {task_name}", 
                        (frame.shape[1] - 340, 60 + i * 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        # Apply the overlay with transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def display_schedule(self, frame):
        """Display schedule information on the frame"""
        if not self.schedule:
            return frame
        
        # Get current time to highlight current schedule item
        current_time = datetime.now().strftime("%H:%M")
        
        # Create a semi-transparent overlay for schedule
        overlay = frame.copy()
        
        # Draw background rectangle
        schedule_height = 40 + 30 * len(self.schedule)
        cv2.rectangle(overlay, (10, frame.shape[0] - schedule_height - 10), 
                     (350, frame.shape[0] - 10), (0, 0, 0), -1)
        
        # Add title
        cv2.putText(overlay, "SCHEDULE:", (20, frame.shape[0] - schedule_height + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display each schedule item
        items = sorted(self.schedule.items(), key=lambda x: x[0])  # Sort by time
        for i, (time_slot, activity) in enumerate(items):
            # Check if this is the current time slot
            is_current = time_slot <= current_time < (items[i+1][0] if i+1 < len(items) else "23:59")
            
            text_color = (0, 255, 255) if is_current else (255, 255, 255)  # Yellow if current, white if not
            
            # Draw schedule text
            cv2.putText(overlay, f"{time_slot} - {activity}", 
                       (20, frame.shape[0] - schedule_height + 40 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        
        # Apply the overlay with transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def run(self):
        try:
            while self.cap.isOpened() and not self.should_exit:
                success, frame = self.cap.read()
                if not success:
                    print("Failed to read from webcam")
                    break
                
                # Flip frame horizontally for a selfie-view
                frame = cv2.flip(frame, 1)
                
                # Convert the BGR image to RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the image with MediaPipe Pose
                pose_results = self.pose.process(image_rgb)
                
                # Detect phone in the frame
                phone_detected, frame = self.detect_phone(frame)
                
                # Get pose state
                pose_state = self.detect_pose_state(pose_results.pose_landmarks)
                
                # Determine the final state
                current_state = self.determine_state(pose_state, phone_detected)
                
                # Update state time if state has changed
                if current_state != self.current_state:
                    self.update_state_time(current_state)
                
                # Draw pose landmarks
                if pose_results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        pose_results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS
                    )
                
                # Display state and times on frame
                state_colors = {
                    "studying": (0, 255, 0),  # Green
                    "not_studying": (0, 165, 255),  # Orange
                    "phone_use": (0, 0, 255)  # Red
                }
                
                # Calculate total session time
                session_time = sum(self.state_times.values())
                if self.prev_state:
                    session_time += (time.time() - self.state_start_time)
                
                # Display state
                cv2.putText(
                    frame, 
                    f"State: {self.current_state}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    state_colors.get(self.current_state, (255, 255, 255)), 
                    2
                )
                
                # Display times
                y_pos = 70
                for state, state_time in self.state_times.items():
                    # Add current state time if this is the active state
                    display_time = state_time
                    if state == self.current_state:
                        display_time += (time.time() - self.state_start_time)
                    
                    cv2.putText(
                        frame,
                        f"{state}: {self.format_time(display_time)}", 
                        (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        state_colors.get(state, (255, 255, 255)), 
                        2
                    )
                    y_pos += 30
                
                # Display session time
                cv2.putText(
                    frame,
                    f"Session: {self.format_time(session_time)}", 
                    (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    2
                )
                
                # Display additional detection information
                cv2.putText(
                    frame,
                    f"Phone detected: {phone_detected}", 
                    (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 255), 
                    2
                )
                
                # Display tasks and schedule
                frame = self.display_tasks(frame)
                frame = self.display_schedule(frame)
                
                # Display the image
                cv2.imshow('Study Monitor', frame)
                
                # Exit on 'q' press
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                    
        finally:
            # Cleanup
            self.should_exit = True
            self.pose.close()
            self.cap.release()
            cv2.destroyAllWindows()
            
            # Wait for server thread to exit
            self.server_thread.join(timeout=1)
            
            # Update final state time
            if self.prev_state:
                end_time = time.time()
                time_in_state = end_time - self.state_start_time
                self.state_times[self.prev_state] += time_in_state
            
            # Export session data to JSON
            self.export_session_data()
    
    def export_session_data(self):
        """Export session data to a JSON file"""
        session_end = datetime.now()
        session_duration = (session_end - self.session_start).total_seconds()
        
        session_data = {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "session_end": session_end.isoformat(),
            "duration_seconds": round(session_duration),
            "stats": {
                "studying": round(self.state_times["studying"]),
                "not_studying": round(self.state_times["not_studying"]),
                "phone_use": round(self.state_times["phone_use"])
            },
            "tasks": self.tasks,
            "completed_tasks": self.completed_tasks,
            "schedule": self.schedule
        }
        
        # Create a filename with the session ID
        filename = f"study_session_{self.session_id}.json"
        
        # Create sessions directory if it doesn't exist
        os.makedirs("sessions", exist_ok=True)
        filepath = os.path.join("sessions", filename)
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=4)
        
        print(f"Session data exported to {filepath}")

def load_session_data(session_id=None):
    """Load session data from a JSON file"""
    if session_id:
        filepath = os.path.join("sessions", f"study_session_{session_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    return None

if __name__ == "__main__":
    print("Starting Study Monitor with YOLOv8...")
    print("Press 'q' to quit and export session data")
    
    parser = argparse.ArgumentParser(description='Study Monitor with Task and Schedule Support')
    parser.add_argument('--session', help='Session ID to resume')
    args = parser.parse_args()
    
    session_data = None
    if args.session:
        session_data = load_session_data(args.session)
    
    if session_data:
        monitor = StudyMonitor(
            session_id=session_data["session_id"],
            tasks=session_data.get("tasks", []),
            schedule=session_data.get("schedule", {})
        )
    else:
        monitor = StudyMonitor()
    
    monitor.run()