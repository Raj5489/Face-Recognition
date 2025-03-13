import cv2
import time

def test_camera():
    print("Testing available cameras...")
    
    # Try different camera indices
    for i in range(4):
        print(f"\nTrying camera index {i}")
        
        # Try different backends
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            try:
                print(f"Trying backend {backend}")
                cap = cv2.VideoCapture(i, backend)
                
                if not cap.isOpened():
                    print(f"Failed to open camera with index {i} and backend {backend}")
                    continue
                
                # Try to read a frame
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to grab frame from camera {i}")
                    cap.release()
                    continue
                
                print(f"Successfully opened camera {i} with backend {backend}")
                print(f"Frame size: {frame.shape}")
                
                # Show the frame
                cv2.imshow(f'Camera Test - Index {i}', frame)
                cv2.waitKey(1000)  # Wait for 1 second
                
                cap.release()
                cv2.destroyAllWindows()
                
                return i, backend
                
            except Exception as e:
                print(f"Error with camera {i} and backend {backend}: {str(e)}")
                if 'cap' in locals():
                    cap.release()
    
    return None, None

if __name__ == "__main__":
    camera_index, backend = test_camera()
    
    if camera_index is not None:
        print(f"\nWorking camera found!")
        print(f"Use these settings in your main program:")
        print(f"Camera Index: {camera_index}")
        print(f"Backend: {backend}")
    else:
        print("\nNo working camera found!")
        print("Please check your webcam connection and drivers.") 