import os
import subprocess
import tempfile
import time
import gc
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

class VRAMManager:
    def __init__(self, device):
        self.device = device
        self.is_cuda = device.type == 'cuda'
        self.max_vram_usage = 0.85  # Use 85% of available VRAM
        
    def get_available_memory(self):
        """Get available VRAM in bytes."""
        if not self.is_cuda:
            return float('inf')  # No limit for CPU
        
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated()
        available = total_memory - allocated_memory
        return int(available * self.max_vram_usage)
    
    def estimate_tensor_memory(self, shape, dtype=torch.float32):
        """Estimate memory usage for a tensor."""
        element_size = torch.tensor([], dtype=dtype).element_size()
        return np.prod(shape) * element_size
    
    def can_fit_batch(self, batch_size, frame_shape):
        """Check if a batch of frames can fit in VRAM."""
        if not self.is_cuda:
            return True
        
        # Estimate memory for batch + intermediate tensors (3x overhead for processing)
        tensor_memory = self.estimate_tensor_memory((batch_size, *frame_shape))
        total_needed = tensor_memory * 3  # Account for intermediate tensors
        
        return total_needed <= self.get_available_memory()
    
    def get_optimal_batch_size(self, frame_shape, max_batch_size=64):
        """Find the largest batch size that fits in VRAM."""
        if not self.is_cuda:
            return max_batch_size
        
        for batch_size in range(max_batch_size, 0, -1):
            if self.can_fit_batch(batch_size, frame_shape):
                return batch_size
        return 1

class ASCIIVideoConverter:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.vram_manager = VRAMManager(self.device)
        self.char_set = DEFAULT_CHARS
        self.font = DEFAULT_FONT
        self.font_size = DEFAULT_FONT_SIZE
        
        # Precompute character brightness levels on GPU
        self.char_brightness = self._precompute_char_brightness()
        
        # Cache for loaded frames (keeps them in VRAM)
        self.frame_cache = {}
        self.cache_size_limit = None  # Will be set based on available VRAM
        
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
    
    def _load_frame_to_vram(self, file_path):
        """Load a single frame directly to VRAM with caching."""
        if file_path in self.frame_cache:
            return self.frame_cache[file_path]
        
        try:
            img = Image.open(file_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert directly to GPU tensor
            tensor = torch.tensor(np.array(img), device=self.device, dtype=torch.float32) / 255.0
            tensor = tensor.permute(2, 0, 1)  # HWC to CHW
            img.close()  # Free PIL memory immediately
            
            # Cache if we have space
            if self.cache_size_limit and len(self.frame_cache) < self.cache_size_limit:
                self.frame_cache[file_path] = tensor
            
            return tensor
            
        except Exception as e:
            print(f"Error loading frame {file_path}: {e}")
            return None
    
    def _load_batch_to_vram(self, file_paths):
        """Load a batch of frames directly to VRAM, with fallback to smaller batches."""
        batch_tensors = []
        failed_files = []
        
        for file_path in file_paths:
            try:
                tensor = self._load_frame_to_vram(file_path)
                if tensor is not None:
                    batch_tensors.append(tensor)
                else:
                    failed_files.append(file_path)
                    
                # Check VRAM usage and break if getting full
                if self.device.type == 'cuda':
                    allocated = torch.cuda.memory_allocated()
                    total = torch.cuda.get_device_properties(0).total_memory
                    if allocated / total > self.vram_manager.max_vram_usage:
                        print(f"VRAM usage high ({allocated/total:.1%}), processing current batch...")
                        break
                        
            except torch.cuda.OutOfMemoryError:
                print("VRAM full, processing current batch...")
                torch.cuda.empty_cache()
                break
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                failed_files.append(file_path)
        
        if batch_tensors:
            try:
                return torch.stack(batch_tensors), failed_files
            except torch.cuda.OutOfMemoryError:
                # If stacking fails, process individually
                print("Batch too large for VRAM, falling back to individual processing")
                torch.cuda.empty_cache()
                return None, file_paths
        
        return None, failed_files
    
    def _frame_to_ascii_tensor_vram(self, frame_tensor, ascii_width):
        """Convert frame tensors to ASCII art tensors using maximum VRAM."""
        try:
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
            
            # GPU-accelerated resize with memory optimization
            grayscale = grayscale.contiguous()  # Ensure memory layout is optimal
            resized = F.interpolate(grayscale.unsqueeze(1), 
                                  size=(ascii_height, ascii_width), 
                                  mode='bilinear',
                                  align_corners=False).squeeze(1)
            
            # Clear intermediate tensor
            del grayscale
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # GPU-accelerated character mapping with memory optimization
            char_brightness = self.char_brightness.view(1, 1, -1)
            
            # Process in chunks if tensor is very large to avoid VRAM overflow
            chunk_size = 1000000  # Process 1M pixels at a time if needed
            total_pixels = resized.shape[1] * resized.shape[2]
            
            if total_pixels > chunk_size:
                # Process in chunks
                B, H, W = resized.shape
                resized_flat = resized.view(B, -1)
                char_indices_flat = torch.zeros_like(resized_flat, dtype=torch.long, device=self.device)
                
                for i in range(0, total_pixels, chunk_size):
                    end_idx = min(i + chunk_size, total_pixels)
                    chunk = resized_flat[:, i:end_idx].unsqueeze(-1)
                    diff = torch.abs(chunk - char_brightness)
                    char_indices_flat[:, i:end_idx] = torch.argmin(diff, dim=-1)
                    del chunk, diff
                
                char_indices = char_indices_flat.view(B, H, W)
                del resized_flat, char_indices_flat
            else:
                # Process all at once
                diff = torch.abs(resized.unsqueeze(-1) - char_brightness)
                char_indices = torch.argmin(diff, dim=-1)
                del diff
            
            del resized
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            return char_indices
            
        except torch.cuda.OutOfMemoryError:
            print("VRAM exhausted during ASCII conversion, falling back to CPU...")
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            # Fallback to CPU processing
            return self._frame_to_ascii_tensor_cpu(frame_tensor.cpu(), ascii_width).to(self.device)
    
    def _frame_to_ascii_tensor_cpu(self, frame_tensor, ascii_width):
        """CPU fallback for ASCII conversion."""
        device = frame_tensor.device
        
        if frame_tensor.ndim == 3:
            frame_tensor = frame_tensor.unsqueeze(0)
        
        # RGB to grayscale
        if frame_tensor.shape[1] == 3:
            grayscale = (0.2989 * frame_tensor[:, 0] + 
                       0.5870 * frame_tensor[:, 1] + 
                       0.1140 * frame_tensor[:, 2])
        else:
            grayscale = frame_tensor.squeeze(1)
        
        # Calculate output dimensions
        B, H, W = grayscale.shape
        ascii_height = int((ascii_width * H / W) * (self.font_size / (self.font_size * 1.8)))
        
        # CPU resize
        grayscale = F.interpolate(grayscale.unsqueeze(1), 
                                size=(ascii_height, ascii_width), 
                                mode='bilinear',
                                align_corners=False).squeeze(1)
        
        # CPU character mapping
        char_brightness_cpu = self.char_brightness.cpu().view(1, 1, -1)
        diff = torch.abs(grayscale.unsqueeze(-1) - char_brightness_cpu)
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
        char_height = int(self.font_size * 1.8)
        
        if output_height is None:
            output_height = ascii_height * char_height
        output_width = ascii_width * char_width
        
        # Convert character indices to CPU only when needed
        char_indices_cpu = char_indices.cpu().numpy()
        
        for frame_idx in range(char_indices.shape[0]):
            img = Image.new('L', (output_width, output_height), color=0)
            draw = ImageDraw.Draw(img)
            
            # Optimized character drawing
            for y in range(ascii_height):
                for x in range(ascii_width):
                    char_idx = char_indices_cpu[frame_idx, y, x]
                    draw.text((x * char_width, y * char_height), 
                            self.char_set[char_idx], fill=255, font=font)
            
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
        batch_size=None,
        progress_callback=None
    ):
        """VRAM-optimized video to ASCII conversion pipeline."""
        start_time_total = time.time()
        
        # Clear any existing cache and free VRAM
        self.frame_cache.clear()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Step 1: Extract frames using FFmpeg
            frame_pattern = temp_path / "frame_%06d.png"
            ffmpeg_cmd = [
                'ffmpeg', '-y',
            ]
            
            # Hardware acceleration
            if torch.cuda.is_available():
                ffmpeg_cmd.extend(['-hwaccel', 'auto'])
            
            if start_time is not None:
                ffmpeg_cmd.extend(['-ss', str(start_time)])
            
            ffmpeg_cmd.extend(['-i', str(input_path)])
            
            if duration is not None:
                ffmpeg_cmd.extend(['-t', str(duration)])
            
            ffmpeg_cmd.extend([
                '-vf', f'fps={fps}' if fps else 'fps=30',
                '-fps_mode', 'cfr',
                str(frame_pattern)
            ])

            # Extract audio separately
            audio_path = temp_path / "audio.aac"
            audio_cmd = [
                'ffmpeg', '-y',
                '-i', str(input_path),
            ]
            if start_time is not None:
                audio_cmd.extend(['-ss', str(start_time)])
            if duration is not None:
                audio_cmd.extend(['-t', str(duration)])
            audio_cmd.extend([
                '-vn',  # No video
                '-acodec', 'aac',
                str(audio_path)
            ])

            # Extract audio (don't fail if no audio exists)
            try:
                subprocess.run(audio_cmd, capture_output=True, text=True, check=True)
                has_audio = True
            except subprocess.CalledProcessError:
                has_audio = False
                if progress_callback:
                    progress_callback("No audio track found, creating video-only output", None)
            
            if progress_callback:
                progress_callback("Extracting frames...", 0)
            
            try:
                subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                error_msg = f"FFmpeg frame extraction failed: {e.stderr}"
                if progress_callback:
                    progress_callback(error_msg, -1)
                raise RuntimeError(error_msg)
            
            # Step 2: Process frames with VRAM optimization
            frame_files = sorted(temp_path.glob("frame_*.png"))
            if not frame_files:
                error_msg = "No frames extracted - check input video"
                if progress_callback:
                    progress_callback(error_msg, -1)
                raise ValueError(error_msg)
            
            # Determine optimal batch size based on first frame and available VRAM
            if batch_size is None:
                sample_img = Image.open(frame_files[0])
                if sample_img.mode != 'RGB':
                    sample_img = sample_img.convert('RGB')
                frame_shape = (3, sample_img.height, sample_img.width)
                batch_size = self.vram_manager.get_optimal_batch_size(frame_shape)
                sample_img.close()
                
                if progress_callback:
                    vram_info = f"Using VRAM-optimized batch size: {batch_size}"
                    if self.device.type == 'cuda':
                        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        vram_info += f" (VRAM: {total_vram:.1f}GB)"
                    progress_callback(vram_info, 5)
            
            # Set cache size based on available memory
            if self.device.type == 'cuda':
                available_mem = self.vram_manager.get_available_memory()
                frame_mem = self.vram_manager.estimate_tensor_memory(frame_shape)
                self.cache_size_limit = min(100, int(available_mem / frame_mem * 0.3))  # Use 30% for cache
            
            ascii_frames = []
            total_frames = len(frame_files)
            processed_frames = 0
            
            # Process in VRAM-optimized batches
            i = 0
            while i < total_frames:
                # Determine actual batch size (might be smaller near the end)
                current_batch_size = min(batch_size, total_frames - i)
                batch_files = frame_files[i:i + current_batch_size]
                
                # Try to load batch to VRAM
                batch_tensor, failed_files = self._load_batch_to_vram(batch_files)
                
                if batch_tensor is not None:
                    # Process entire batch in VRAM
                    try:
                        char_indices = self._frame_to_ascii_tensor_vram(batch_tensor, ascii_width)
                        ascii_frames.extend(self._render_ascii_frames(char_indices))
                        processed_frames += batch_tensor.shape[0]
                        
                        # Clear VRAM
                        del batch_tensor, char_indices
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"Batch processing failed: {e}, falling back to individual frames")
                        # Process frames individually
                        for file_path in batch_files:
                            try:
                                tensor = self._load_frame_to_vram(file_path)
                                if tensor is not None:
                                    char_indices = self._frame_to_ascii_tensor_vram(tensor.unsqueeze(0), ascii_width)
                                    ascii_frames.extend(self._render_ascii_frames(char_indices))
                                    processed_frames += 1
                                    del tensor, char_indices
                            except Exception as inner_e:
                                print(f"Failed to process {file_path}: {inner_e}")
                else:
                    # VRAM full, process smaller batches or individual frames
                    smaller_batch_size = max(1, current_batch_size // 2)
                    if smaller_batch_size < current_batch_size:
                        # Try smaller batch
                        continue
                    else:
                        # Process individual frames
                        for file_path in batch_files:
                            try:
                                tensor = self._load_frame_to_vram(file_path)
                                if tensor is not None:
                                    char_indices = self._frame_to_ascii_tensor_vram(tensor.unsqueeze(0), ascii_width)
                                    ascii_frames.extend(self._render_ascii_frames(char_indices))
                                    processed_frames += 1
                                    del tensor, char_indices
                                    if self.device.type == 'cuda':
                                        torch.cuda.empty_cache()
                            except Exception as e:
                                print(f"Failed to process {file_path}: {e}")
                
                i += current_batch_size
                
                # Update progress
                progress = (processed_frames / total_frames) * 90  # Reserve 10% for encoding
                if progress_callback:
                    vram_usage = ""
                    if self.device.type == 'cuda':
                        allocated = torch.cuda.memory_allocated()
                        total_mem = torch.cuda.get_device_properties(0).total_memory
                        vram_usage = f" (VRAM: {allocated/total_mem:.1%})"
                    progress_callback(f"Processing {processed_frames}/{total_frames} frames{vram_usage}", progress)
            
            # Clear all caches before encoding
            self.frame_cache.clear()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            # Step 3: Encode final video
            if progress_callback:
                progress_callback("Encoding final video...", 95)
            
            ascii_frame_pattern = temp_path / "ascii_frame_%06d.png"
            for i, frame in enumerate(ascii_frames):
                frame.save(str(ascii_frame_pattern) % (i + 1))
            
            # FFmpeg encoding
            # FFmpeg encoding with audio
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps) if fps else '30',
                '-i', str(ascii_frame_pattern),
            ]

            # Add audio if it exists
            if has_audio and (temp_path / "audio.aac").exists():
                ffmpeg_cmd.extend(['-i', str(temp_path / "audio.aac")])
                ffmpeg_cmd.extend([
                    '-c:v', 'libx264',
                    '-c:a', 'aac',
                    '-preset', 'medium',
                    '-crf', '18',
                    '-pix_fmt', 'yuv420p',
                    '-shortest'  # Match shortest stream duration
                ])
            else:
                ffmpeg_cmd.extend([
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-crf', '18',
                    '-pix_fmt', 'yuv420p'
                ])

            ffmpeg_cmd.append(str(output_path))
            
            try:
                subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                error_msg = f"Video encoding failed: {e.stderr}"
                if progress_callback:
                    progress_callback(error_msg, -1)
                raise RuntimeError(error_msg)
        
        elapsed = time.time() - start_time_total
        audio_status = "with audio" if has_audio else "video only"
        success_msg = (f"Success! Saved to: {output_path}\n"
                    f"Resolution: {ascii_width} chars wide\n"
                    f"Processed: {processed_frames} frames ({audio_status})\n"
                    f"Time: {elapsed:.2f}s (~{processed_frames/elapsed:.1f} FPS)")
        
        if progress_callback:
            progress_callback(success_msg, 100)
        
        return success_msg

class ASCIIVideoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VRAM-Optimized ASCII Video Converter")
        self.root.geometry("800x650")
        
        # Converter instance
        self.converter = ASCIIVideoConverter()
        
        # Variables
        self.input_path = StringVar()
        self.output_path = StringVar()
        self.ascii_width = IntVar(value=100)
        self.fps = IntVar(value=30)
        self.start_time = DoubleVar(value=0)
        self.duration = DoubleVar(value=10)
        self.auto_batch = BooleanVar(value=True)
        self.manual_batch_size = IntVar(value=16)
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
        
        # VRAM optimization settings
        vram_frame = ttk.Frame(settings_frame)
        vram_frame.grid(row=4, column=0, columnspan=2, sticky=EW, pady=5)
        
        ttk.Checkbutton(vram_frame, text="Auto-optimize VRAM batch size", 
                       variable=self.auto_batch, command=self.toggle_batch_mode).pack(side=LEFT)
        
        self.manual_batch_frame = ttk.Frame(vram_frame)
        self.manual_batch_frame.pack(side=RIGHT)
        ttk.Label(self.manual_batch_frame, text="Manual batch size:").pack(side=LEFT)
        ttk.Spinbox(self.manual_batch_frame, from_=1, to=64, 
                   textvariable=self.manual_batch_size, width=10).pack(side=LEFT)
        
        # VRAM info
        self.vram_info_label = ttk.Label(settings_frame, text="", foreground="blue")
        self.vram_info_label.grid(row=5, column=0, columnspan=2, sticky=W)
        self.update_vram_info()
        
        # Preview button
        ttk.Button(settings_frame, text="Preview Settings", command=self.preview_settings).grid(row=6, column=0, columnspan=2, pady=5)
        
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
        
        # Initialize batch mode
        self.toggle_batch_mode()
    
    def update_vram_info(self):
        """Update VRAM information display."""
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_vram = props.total_memory / 1024**3
            allocated = torch.cuda.memory_allocated() / 1024**3
            self.vram_info_label.config(
                text=f"GPU: {props.name} | VRAM: {allocated:.1f}/{total_vram:.1f} GB"
            )
        else:
            self.vram_info_label.config(text="CUDA not available - using CPU")
    
    def toggle_batch_mode(self):
        """Toggle between auto and manual batch size."""
        if self.auto_batch.get():
            for child in self.manual_batch_frame.winfo_children():
                child.config(state='disabled')
        else:
            for child in self.manual_batch_frame.winfo_children():
                child.config(state='normal')
    
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
                    'ffmpeg', '-y',
                    '-ss', str(self.start_time.get()),
                    '-i', self.input_path.get(),
                    '-vframes', '1',
                    str(frame_path)
                ]
                
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                
                if not frame_path.exists():
                    raise RuntimeError("Failed to extract preview frame")
                
                # Load frame and convert to ASCII using VRAM
                tensor = self.converter._load_frame_to_vram(frame_path)
                if tensor is None:
                    raise RuntimeError("Failed to load frame to VRAM")
                
                # Convert to ASCII
                char_indices = self.converter._frame_to_ascii_tensor_vram(tensor.unsqueeze(0), self.ascii_width.get())
                ascii_frames = self.converter._render_ascii_frames(char_indices, output_height=300)
                
                # Show preview in new window
                preview_window = Toplevel(self.root)
                preview_window.title("ASCII Preview")
                preview_window.geometry("600x400")
                
                # Convert PIL Image to PhotoImage
                ascii_image = ascii_frames[0]
                # Resize for display if too large
                display_image = ascii_image.copy()
                if display_image.width > 500 or display_image.height > 350:
                    display_image.thumbnail((500, 350), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(display_image)
                
                label = ttk.Label(preview_window, image=photo)
                label.image = photo  # Keep reference
                label.pack(padx=10, pady=10)
                
                info_text = f"ASCII Size: {char_indices.shape[2]}x{char_indices.shape[1]} characters"
                ttk.Label(preview_window, text=info_text).pack(pady=5)
                
                ttk.Button(preview_window, text="Close", command=preview_window.destroy).pack(pady=5)
                
                # Clean up VRAM
                del tensor, char_indices
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
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
        
        # VRAM warning if not available
        if not torch.cuda.is_available():
            if not messagebox.askokcancel(
                "Warning", 
                "CUDA is not available. Conversion will run on CPU and may be much slower. Continue?"
            ):
                return
        
        # Disable controls during processing
        self.is_processing = True
        self.progress_bar['value'] = 0
        self.log_message("Starting VRAM-optimized conversion...")
        self.update_vram_info()
        
        # Run conversion in a separate thread
        import threading
        thread = threading.Thread(target=self.run_conversion, daemon=True)
        thread.start()
    
    def run_conversion(self):
        try:
            # Determine batch size
            batch_size = None if self.auto_batch.get() else self.manual_batch_size.get()
            
            result = self.converter.convert_video_to_ascii(
                input_path=self.input_path.get(),
                output_path=self.output_path.get(),
                ascii_width=self.ascii_width.get(),
                fps=self.fps.get(),
                start_time=self.start_time.get(),
                duration=self.duration.get(),
                batch_size=batch_size,
                progress_callback=self.update_progress
            )
            
            self.log_message(result)
            messagebox.showinfo("Success", "Video conversion completed successfully!")
            
        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Conversion failed:\n{str(e)}")
            
        finally:
            self.is_processing = False
            self.root.after(0, self.update_vram_info)
    
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
        self.update_vram_info()
    
    def log_message(self, message):
        self.log_text.insert(END, message + "\n")
        self.log_text.see(END)
        self.log_text.update()

if __name__ == "__main__":
    root = Tk()
    app = ASCIIVideoGUI(root)
    root.mainloop()