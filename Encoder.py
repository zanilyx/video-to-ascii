import os
import subprocess
import tempfile
import time
from pathlib import Path
from tkinter import *
from tkinter import ttk, filedialog, messagebox
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

# Configuration
DEFAULT_CHARS = "@%#*+=-:. "  # Simplified character set for better contrast
DEFAULT_FONT = "Courier.ttf"
DEFAULT_FONT_SIZE = 10  # Smaller font for better ASCII art resolution

class ASCIIVideoConverter:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.char_set = DEFAULT_CHARS
        self.font = DEFAULT_FONT
        self.font_size = DEFAULT_FONT_SIZE
        
        # Precompute character brightness levels on GPU
        self.char_brightness = self._precompute_char_brightness()
        
    def _precompute_char_brightness(self):
        """Precompute brightness values for each character in the set using GPU."""
        # Create a blank image to measure character brightness
        img = Image.new('L', (self.font_size, self.font_size), color=0)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(self.font, self.font_size)
        except:
            font = ImageFont.load_default()
        
        brightness = []
        for char in self.char_set:
            # Clear the image
            draw.rectangle((0, 0, self.font_size, self.font_size), fill=0)
            # Draw the character
            draw.text((0, 0), char, fill=255, font=font)
            # Calculate brightness (normalized 0-1)
            brightness.append(np.array(img).mean() / 255.0)
        
        return torch.tensor(brightness, device=self.device, dtype=torch.float32)
    
    def _frame_to_ascii_tensor(self, frame_tensor, ascii_width):
        """Convert frame tensors to ASCII art tensors using GPU ops."""
        if frame_tensor.ndim == 3:
            frame_tensor = frame_tensor.unsqueeze(0)
        
        # RGB to grayscale using GPU-optimized weights
        if frame_tensor.shape[1] == 3:
            grayscale = (0.2989 * frame_tensor[:, 0] + 
                         0.5870 * frame_tensor[:, 1] + 
                         0.1140 * frame_tensor[:, 2])
        else:
            grayscale = frame_tensor.squeeze(1)
        
        # Calculate output dimensions maintaining aspect ratio
        B, H, W = grayscale.shape
        ascii_height = int((ascii_width * H / W) * (self.font_size / (self.font_size * 1.8)))
        
        # GPU-accelerated resize
        grayscale = F.interpolate(grayscale.unsqueeze(1), 
                                size=(ascii_height, ascii_width), 
                                mode='bilinear',
                                align_corners=False).squeeze(1)
        
        # GPU-accelerated character mapping
        char_brightness = self.char_brightness.view(1, 1, -1)
        diff = torch.abs(grayscale.unsqueeze(-1) - char_brightness)
        char_indices = torch.argmin(diff, dim=-1)
        
        return char_indices
    
    def _render_ascii_frames(self, char_indices, output_height=None):
        """Render ASCII frames using optimized PIL operations."""
        frames = []
        try:
            font = ImageFont.truetype(self.font, self.font_size)
        except:
            font = ImageFont.load_default()
        
        ascii_height, ascii_width = char_indices.shape[1], char_indices.shape[2]
        char_width = self.font_size
        char_height = int(self.font_size * 1.8)  # Adjusted for better aspect ratio
        
        if output_height is None:
            output_height = ascii_height * char_height
        output_width = ascii_width * char_width
        
        # Convert all character indices to CPU at once
        char_indices_cpu = char_indices.cpu().numpy()
        
        for frame_idx in range(char_indices.shape[0]):
            img = Image.new('L', (output_width, output_height), color=0)
            draw = ImageDraw.Draw(img)
            
            # Pre-calculate positions
            positions = [(x * char_width, y * char_height) 
                        for y in range(ascii_height) 
                        for x in range(ascii_width)]
            
            # Draw all characters in one pass
            for (x, y), char_idx in zip(positions, char_indices_cpu[frame_idx].flatten()):
                draw.text((x, y), self.char_set[char_idx], fill=255, font=font)
            
            frames.append(img)
        
        return frames
    
    def convert_video_to_ascii(
        self,
        input_path,
        output_path,
        ascii_width=100,
        fps=None,
        start_time=None,
        duration=None,
        batch_size=16,
        progress_callback=None
    ):
        """Full GPU-accelerated video to ASCII conversion pipeline."""
        start_time_total = time.time()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Step 1: Extract frames using CPU FFmpeg (more reliable)
            frame_pattern = temp_path / "frame_%06d.png"
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',
            ]
            
            # Add hardware acceleration if available, but fallback gracefully
            if torch.cuda.is_available():
                # Try CUDA acceleration but don't force it
                ffmpeg_cmd.extend(['-hwaccel', 'auto'])
            
            if start_time is not None:
                ffmpeg_cmd.extend(['-ss', str(start_time)])
            
            ffmpeg_cmd.extend(['-i', str(input_path)])
            
            if duration is not None:
                ffmpeg_cmd.extend(['-t', str(duration)])
            
            # Use CPU-based frame extraction for reliability
            ffmpeg_cmd.extend([
                '-vf', f'fps={fps}' if fps else 'fps=30',
                '-fps_mode', 'cfr',  # Replace deprecated -vsync
                str(frame_pattern)
            ])
            
            if progress_callback:
                progress_callback("Extracting frames...", 0)
            
            try:
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                error_msg = f"FFmpeg frame extraction failed: {e.stderr}"
                if progress_callback:
                    progress_callback(error_msg, -1)
                raise RuntimeError(error_msg)
            
            # Step 2: Process frames in GPU batches
            frame_files = sorted(temp_path.glob("frame_*.png"))
            if not frame_files:
                error_msg = "No frames extracted - check input video"
                if progress_callback:
                    progress_callback(error_msg, -1)
                raise ValueError(error_msg)
            
            ascii_frames = []
            total_frames = len(frame_files)
            for i in range(0, total_frames, batch_size):
                batch_files = frame_files[i:i+batch_size]
                
                # Batch load frames directly to GPU
                batch_tensors = []
                for file in batch_files:
                    try:
                        img = Image.open(file)
                        # Convert to RGB if needed
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        tensor = torch.tensor(np.array(img), device=self.device, dtype=torch.float32) / 255.0
                        tensor = tensor.permute(2, 0, 1)  # HWC to CHW
                        batch_tensors.append(tensor)
                        img.close()  # Free memory immediately
                    except Exception as e:
                        if progress_callback:
                            progress_callback(f"Error loading frame {file}: {e}", -1)
                        continue
                
                if not batch_tensors:
                    continue
                
                batch_tensor = torch.stack(batch_tensors)
                
                # GPU-accelerated ASCII conversion
                char_indices = self._frame_to_ascii_tensor(batch_tensor, ascii_width)
                ascii_frames.extend(self._render_ascii_frames(char_indices))
                
                # Update progress
                progress = (i + len(batch_files)) / total_frames * 90  # Reserve 10% for encoding
                if progress_callback:
                    progress_callback(f"Processing {i + len(batch_files)}/{total_frames} frames", progress)
            
            # Step 3: Encode final video (CPU-based for reliability)
            if progress_callback:
                progress_callback("Encoding final video...", 95)
            
            ascii_frame_pattern = temp_path / "ascii_frame_%06d.png"
            for i, frame in enumerate(ascii_frames):
                frame.save(str(ascii_frame_pattern) % (i + 1))
            
            # Use reliable CPU encoding
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',
                '-framerate', str(fps) if fps else '30',
                '-i', str(ascii_frame_pattern),
                '-c:v', 'libx264',  # Use CPU encoder for reliability
                '-preset', 'medium',
                '-crf', '18',  # Good quality
                '-pix_fmt', 'yuv420p',
                str(output_path)
            ]
            
            try:
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                error_msg = f"Video encoding failed: {e.stderr}"
                if progress_callback:
                    progress_callback(error_msg, -1)
                raise RuntimeError(error_msg)
        
        elapsed = time.time() - start_time_total
        success_msg = (f"Success! Saved to: {output_path}\n"
                      f"Resolution: {ascii_width} chars wide\n"
                      f"Time: {elapsed:.2f}s (~{total_frames/elapsed:.1f} FPS)")
        
        if progress_callback:
            progress_callback(success_msg, 100)
        
        return success_msg

class ASCIIVideoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GPU Accelerated ASCII Video Converter")
        self.root.geometry("800x600")
        
        # Converter instance
        self.converter = ASCIIVideoConverter()
        
        # Variables
        self.input_path = StringVar()
        self.output_path = StringVar()
        self.ascii_width = IntVar(value=100)
        self.fps = IntVar(value=30)
        self.start_time = DoubleVar(value=0)
        self.duration = DoubleVar(value=10)
        self.batch_size = IntVar(value=16)
        self.is_processing = False
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=BOTH, expand=True)
        
        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Input Video", padding="10")
        input_frame.pack(fill=X, pady=5)
        
        ttk.Label(input_frame, text="Input File:").grid(row=0, column=0, sticky=W)
        ttk.Entry(input_frame, textvariable=self.input_path, width=50).grid(row=0, column=1, sticky=EW)
        ttk.Button(input_frame, text="Browse...", command=self.browse_input).grid(row=0, column=2, padx=5)
        
        # Output section
        output_frame = ttk.LabelFrame(main_frame, text="Output Video", padding="10")
        output_frame.pack(fill=X, pady=5)
        
        ttk.Label(output_frame, text="Output File:").grid(row=0, column=0, sticky=W)
        ttk.Entry(output_frame, textvariable=self.output_path, width=50).grid(row=0, column=1, sticky=EW)
        ttk.Button(output_frame, text="Browse...", command=self.browse_output).grid(row=0, column=2, padx=5)
        
        # Settings section
        settings_frame = ttk.LabelFrame(main_frame, text="Conversion Settings", padding="10")
        settings_frame.pack(fill=X, pady=5)
        
        # ASCII width
        ttk.Label(settings_frame, text="ASCII Width (characters):").grid(row=0, column=0, sticky=W)
        ttk.Spinbox(settings_frame, from_=20, to=300, textvariable=self.ascii_width).grid(row=0, column=1, sticky=W)
        
        # FPS
        ttk.Label(settings_frame, text="Output FPS:").grid(row=1, column=0, sticky=W)
        ttk.Spinbox(settings_frame, from_=1, to=60, textvariable=self.fps).grid(row=1, column=1, sticky=W)
        
        # Start time
        ttk.Label(settings_frame, text="Start Time (seconds):").grid(row=2, column=0, sticky=W)
        ttk.Spinbox(settings_frame, from_=0, to=1000, increment=1, textvariable=self.start_time).grid(row=2, column=1, sticky=W)
        
        # Duration
        ttk.Label(settings_frame, text="Duration (seconds):").grid(row=3, column=0, sticky=W)
        ttk.Spinbox(settings_frame, from_=1, to=1000, increment=1, textvariable=self.duration).grid(row=3, column=1, sticky=W)
        
        # Batch size
        ttk.Label(settings_frame, text="Batch Size:").grid(row=4, column=0, sticky=W)
        ttk.Spinbox(settings_frame, from_=1, to=64, textvariable=self.batch_size).grid(row=4, column=1, sticky=W)
        
        # Preview button
        ttk.Button(settings_frame, text="Preview Settings", command=self.preview_settings).grid(row=5, column=0, columnspan=2, pady=5)
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.pack(fill=BOTH, expand=True, pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="Ready to convert...")
        self.progress_label.pack(fill=X, pady=5)
        
        self.progress_bar = ttk.Progressbar(progress_frame, orient=HORIZONTAL, mode='determinate')
        self.progress_bar.pack(fill=X, pady=5)
        
        # Log output
        self.log_text = Text(progress_frame, height=8, wrap=WORD)
        self.log_text.pack(fill=BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(progress_frame, orient=VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.log_text['yscrollcommand'] = scrollbar.set
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=X, pady=5)
        
        ttk.Button(button_frame, text="Convert", command=self.start_conversion).pack(side=LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel_conversion).pack(side=LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=RIGHT, padx=5)
        
        # Configure grid weights
        input_frame.columnconfigure(1, weight=1)
        output_frame.columnconfigure(1, weight=1)
        settings_frame.columnconfigure(1, weight=1)
        
    def browse_input(self):
        file_path = filedialog.askopenfilename(
            title="Select Input Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if file_path:
            self.input_path.set(file_path)
            # Set default output path
            dir_name, file_name = os.path.split(file_path)
            name, ext = os.path.splitext(file_name)
            self.output_path.set(os.path.join(dir_name, f"{name}_ascii.mp4"))
    
    def browse_output(self):
        file_path = filedialog.asksaveasfilename(
            title="Save Output Video",
            defaultextension=".mp4",
            filetypes=[("MP4 Files", "*.mp4"), ("All Files", "*.*")]
        )
        if file_path:
            self.output_path.set(file_path)
    
    def preview_settings(self):
        """Show a preview of the current settings."""
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input video file first.")
            return
        
        try:
            # Extract a single frame for preview
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                frame_path = temp_path / "preview_frame.png"
                
                # Extract frame at start time
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-ss', str(self.start_time.get()),
                    '-i', self.input_path.get(),
                    '-vframes', '1',
                    str(frame_path)
                ]
                
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                
                if not frame_path.exists():
                    raise RuntimeError("Failed to extract preview frame")
                
                # Load frame and convert to ASCII
                img = Image.open(frame_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                tensor = torch.tensor(np.array(img), device=self.converter.device, dtype=torch.float32) / 255.0
                tensor = tensor.permute(2, 0, 1)  # HWC to CHW
                
                # Convert to ASCII
                char_indices = self.converter._frame_to_ascii_tensor(tensor, self.ascii_width.get())
                ascii_frames = self.converter._render_ascii_frames(char_indices, output_height=300)
                
                # Show preview in new window
                preview_window = Toplevel(self.root)
                preview_window.title("ASCII Preview")
                
                # Convert PIL Image to PhotoImage
                ascii_image = ascii_frames[0]
                photo = ImageTk.PhotoImage(ascii_image)
                
                label = ttk.Label(preview_window, image=photo)
                label.image = photo  # Keep reference
                label.pack(padx=10, pady=10)
                
                ttk.Button(preview_window, text="Close", command=preview_window.destroy).pack(pady=5)
                
        except Exception as e:
            messagebox.showerror("Preview Error", f"Failed to generate preview:\n{str(e)}")
    
    def start_conversion(self):
        if self.is_processing:
            return
        
        # Validate inputs
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input video file.")
            return
        
        if not self.output_path.get():
            messagebox.showerror("Error", "Please select an output video file.")
            return
        
        if not os.path.exists(self.input_path.get()):
            messagebox.showerror("Error", "Input video file does not exist.")
            return
        
        # Check FFmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except:
            messagebox.showerror("Error", "FFmpeg is not installed or not in PATH. Please install FFmpeg.")
            return
        
        # Check CUDA is available
        if not torch.cuda.is_available():
            if not messagebox.askokcancel(
                "Warning", 
                "CUDA is not available. Conversion will run on CPU and may be slow. Continue?"
            ):
                return
        
        # Disable controls during processing
        self.is_processing = True
        self.progress_bar['value'] = 0
        self.log_message("Starting conversion...")
        
        # Run conversion in a separate thread
        import threading
        thread = threading.Thread(target=self.run_conversion, daemon=True)
        thread.start()
    
    def run_conversion(self):
        try:
            result = self.converter.convert_video_to_ascii(
                input_path=self.input_path.get(),
                output_path=self.output_path.get(),
                ascii_width=self.ascii_width.get(),
                fps=self.fps.get(),
                start_time=self.start_time.get(),
                duration=self.duration.get(),
                batch_size=self.batch_size.get(),
                progress_callback=self.update_progress
            )
            
            self.log_message(result)
            messagebox.showinfo("Success", "Video conversion completed successfully!")
            
        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Conversion failed:\n{str(e)}")
            
        finally:
            self.is_processing = False
    
    def cancel_conversion(self):
        if self.is_processing:
            if messagebox.askyesno("Confirm", "Are you sure you want to cancel the conversion?"):
                # TODO: Implement proper cancellation
                self.is_processing = False
                self.log_message("Conversion cancelled by user.")
    
    def update_progress(self, message, percent):
        self.root.after(0, self._update_progress_gui, message, percent)
    
    def _update_progress_gui(self, message, percent):
        self.log_message(message)
        if percent >= 0:
            self.progress_bar['value'] = percent
        else:
            self.progress_bar['value'] = 0
    
    def log_message(self, message):
        self.log_text.insert(END, message + "\n")
        self.log_text.see(END)
        self.log_text.update()

if __name__ == "__main__":
    root = Tk()
    app = ASCIIVideoGUI(root)
    root.mainloop()