import difflib
import musicbrainzngs
import os
import re
import requests
import subprocess
import wave
from pytubefix import YouTube, Playlist
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3, APIC
import threading
import concurrent.futures
import time
import random
from urllib.parse import parse_qs, urlparse

APP_NAME = "AlbumWorks"
APP_VERSION = "1.0.4"
APP_URL = "https://github.com/mtalyat/AlbumWorks"

FFMPEG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "ffmpeg", "bin", "ffmpeg.exe")
OUTPUT_PATH = os.path.join(os.path.expanduser("~"), "Music")
FORMATS = ["mp3", "wav", "mp4a"]
TEMP_NAME = "temp"
THUMBNAIL_NAME = "thumbnail"
UNKNOWN = "Unknown"
TRIM_SILENCE = True
TRIM_SILENCE_THRESHOLD = "-50dB"
TRIM_SILENCE_MIN_DURATION = "0.25"
YT_PO_TOKEN_ENV = "ALBUMWORKS_YT_PO_TOKEN"
YT_VISITOR_DATA_ENV = "ALBUMWORKS_YT_VISITOR_DATA"
YOUTUBE_REQUEST_SPACING_SECONDS = 0.5

ARTWORK_TYPES = ['file', 'url', 'thumbnail']
ARTWORK_TYPE_FILE = 0 # file on disk
ARTWORK_TYPE_LINK = 1 # link to an online image
ARTWORK_TYPE_THUMBNAIL = 2 # YouTube thumbnail

COLOR_ERROR = "\033[91m"
COLOR_SUCCESS = "\033[92m"
COLOR_INFO = "\033[90m"
COLOR_RESET = "\033[0m"

RESULT_SUCCESS = "SUCCESS"
RESULT_SKIPPED = "SKIPPED"
RESULT_ERROR = "ERROR"

class DownloadResultDisplay:
    def __init__(self):
        self.lock = threading.Lock()
        self.results = []

    def reset(self):
        with self.lock:
            self.results = []

    def add(self, status, item, detail = None, track_number = None):
        with self.lock:
            self.results.append((status, item, detail, track_number))

    def print_results(self):
        with self.lock:
            if not self.results:
                return
            snapshot = list(self.results)

        print()
        for status, item, detail, track_number in snapshot:
            color = COLOR_INFO
            if status == RESULT_SUCCESS:
                color = COLOR_SUCCESS
            elif status == RESULT_ERROR:
                color = COLOR_ERROR
            
            # If succeeded, it has been written to a file
            # If skipped, the file already exists
            # If failed, we will want to know why
            if status == RESULT_ERROR:
                suffix = f" -> {detail}" if detail else ""
            else:
                suffix = ''  # No suffix for success or skipped
            track_prefix = f"{track_number}. " if track_number is not None else ""
            print(f"{color}[{status}] {track_prefix}{item}{suffix}{COLOR_RESET}")

    def print_summary(self):
        with self.lock:
            if not self.results:
                return
            success = sum(1 for status, _, _, _ in self.results if status == RESULT_SUCCESS)
            skipped = sum(1 for status, _, _, _ in self.results if status == RESULT_SKIPPED)
            errors = sum(1 for status, _, _, _ in self.results if status == RESULT_ERROR)
        if skipped + errors > 0:
            print(f"Results: {success} success, {skipped} skipped, {errors} errors.")

g_result_display = DownloadResultDisplay()

def add_download_result(status, item, detail = None, track_number = None):
    global g_result_display
    g_result_display.add(status, item, detail, track_number)

class WorkerTask:
    def __init__(self, priority):
        self.priority = priority
        self.progress = 0

    def update(self, progress, weight):
        self.progress = min(self.progress + progress, weight)

    def set(self, progress):
        self.progress = progress

class Worker:
    def __init__(self, task_weight, task_count, width=50):
        self.total = task_count * task_weight
        self.task_count = task_count
        self.task_weight = task_weight
        self.lock = threading.Lock()
        self.tasks: dict[str, WorkerTask] = dict()
        self.width = width

        self.display()

    def _check_priority(self, old, new):
        return old < new
    
    def abort_task(self):
        if self.task_count <= 0:
            return
        with self.lock:
            self.task_count -= 1
            self.total -= self.task_weight

    def start_task(self, id, priority) -> bool:
        self.wait_for_task(id)
        with self.lock:
            if id in self.tasks:
                if not self._check_priority(self.tasks[id].priority, priority):
                    return False
            self.tasks[id] = WorkerTask(priority)
            assert len(self.tasks) <= self.task_count
            return True
        
    def cancel_task(self, id):
        if id is None: return
        with self.lock:
            if not id in self.tasks:
                self.tasks[id] = WorkerTask(0)
            self.tasks[id].set(self.task_weight)
        self.display()
        
    def wait_for_task(self, id):
        while True:
            with self.lock:
                if id not in self.tasks:
                    return
                if self.tasks[id].progress >= self.task_weight:
                    return
            threading.Event().wait(0.1)

    def update(self, task_id, progress):
        with self.lock:
            if task_id not in self.tasks:
                return
            self.tasks[task_id].update(progress, self.task_weight)
        self.display()

    def increment(self, task_id):
        self.update(task_id, 1)

    def get_total(self):
        return self.total
    
    def get_current(self):
        with self.lock:
            return sum(task.progress for task in self.tasks.values())

    def display(self):
        total = self.get_total()
        current = self.get_current()
        if current > total:
            current = total
        if current == total:
            print(f"\rProgress: [{'#' * self.width}] 100%")
            return
        percent = (current / total * 100) if total else 0
        bar_length = int(self.width * percent / 100)
        bar = "#" * bar_length + "-" * (self.width - bar_length)
        print(f"\rProgress: [{bar}] {percent:.0f}%", end="")

def increment_or_add_suffix(string):
    """
    Adds a "2" to the end of the string if it doesn't end with a number,
    or increments the number at the end by 1 if it does.

    Args:
        string (str): The input string.

    Returns:
        str: The modified string.
    """
    # Match a number at the end of the string
    match = re.search(r' (\d+)$', string)
    if match:
        # Increment the number
        number = int(match.group(1))
        incremented_number = number + 1
        return string[:match.start()] + str(incremented_number)
    else:
        # Add "2" to the end
        return string + " 2"

def fix_for_path(path):
    # Remove illegal characters for file and folder names
    path = re.sub(r'[<>:"/\\|?*]', '', path)
    return path.strip()

class Title:
    display_name: str
    file_name: str
    track_index: int

    def __init__(self, title):
        self.set_title(title)
    
    def __repr__(self):
        return self.display_name
    
    def set_title(self, title, index = -1):
        self.display_name = title
        self.file_name = fix_for_path(title)
        self.track_index = index
    
    def fix_title(self, album = None, readonly = True):
        """
        Cleans up the YouTube title by removing unwanted text and illegal characters.

        Args:
            title (str): The original YouTube title.

        Returns:
            str: The sanitized title.
        """
        title = self.display_name

        # Remove "(...)"/"[...]" text"
        title = re.sub(r'(\[.*?\]|\(.*?\))', '', title, flags=re.IGNORECASE)

        # Remove any common words or phrases "OST"
        title = re.sub(r'\b(OST|Lyrics|(Full )?Album)\b', '', title, flags=re.IGNORECASE)

        # Remove hz or kHz
        title = re.sub(r'\b(\d{2,4}\s?(hz|khz))\b', '', title, flags=re.IGNORECASE)

        # Remove anything past '|'
        title = re.sub(r'\|.*$', '', title, flags=re.IGNORECASE)

        # If there is a quote block, return that
        quote_pattern = r'[\'"](.+?)[\'"]'
        quote_match = re.search(quote_pattern, title)
        if quote_match:
            title = quote_match.group(1)

        # Remove any numbers at the start of the title
        title = re.sub(r'^\d+[).]?\s*', '', title, flags=re.IGNORECASE)

        # Remove " - "or " | ", etc.
        title = re.sub(r' ?[-|] ?', '', title)

        # Remove leading or trailing whitespace and punctuation
        title = re.sub(r'^[^a-zA-Z0-9\(]+|[^a-zA-Z0-9\)]+$', '', title)

        # If no album, done here
        if not album:
            self.set_title(title)
            return
        
        # Remove any mention of the artist name
        title = re.sub(rf'{album.artist}', '', title, flags=re.IGNORECASE).strip()

        # Remove any mention of the album name
        temp = re.sub(rf'{album.name}', '', title, flags=re.IGNORECASE).strip()

        # If the title is empty, use the album name
        empty_pattern = r'^[^a-zA-Z0-9]*$'
        if re.fullmatch(empty_pattern, temp):
            title = album.name
        else:
            title = temp

        # If there is a match to a track, use that instead
        track_index = album.get_track_index(title, readonly)
        if track_index != -1:
            title = album.tracks[track_index]

        self.set_title(title, track_index)

def find_closest_string_index(target: str, string_list, used: list[bool]) -> int:
    """
    Finds the index of the closest string in a list to the given target string.

    Args:
        target (str): The string to compare against.
        string_list (list): A list of strings to search.

    Returns:
        int: The index of the closest matching string, or -1 if no match is found.
    """
    def _find_closest_string_index(target: str, string_list, used: list[bool]) -> int:
        # try matching the first string that starts with the target
        for i, string in enumerate(string_list):
            if len(string) == 0:
                continue
            if string.startswith(target) and not used[i]:
                return i
        # try the opposite
        for i, string in enumerate(string_list):
            if len(string) == 0:
                continue
            if string in target and not used[i]:
                return i
        # try matching closest string using difflib
        matches = difflib.get_close_matches(target, string_list, n=1, cutoff=0.8)
        if matches:
            closest_match = matches[0]
            if len(closest_match) == 0:
                return -1
            i = string_list.index(closest_match)
            if not used[i]:
                return i
        return -1
    
    # replace fancy quotes with normal quotes
    target = re.sub(r'[‘’]', '\'', re.sub(r'[“”]', '"', target)).lower()
    string_list = [re.sub(r'[‘’]', '\'', re.sub(r'[“”]', '"', s)).lower() for s in string_list]  # Normalize case for comparison
    
    index = _find_closest_string_index(target, string_list, used)
    if index != -1:
        return index
    
    # try again but remove all (...) and [...]
    target_cleaned = re.sub(r'(\[.*?\]|\(.*?\))', '', target).strip()
    string_list_cleaned = [re.sub(r'(\[.*?\]|\(.*?\))', '', s).strip() for s in string_list]
    index = _find_closest_string_index(target_cleaned, string_list_cleaned, used)
    if index != -1:
        return index
    
    # not found
    return -1

class Album:
    def __init__(self, name, artist, year, genre, tracks, artwork):
        self.name = name
        self.artist = artist
        self.year = year
        self.genre = genre
        self.artwork = artwork
        self.tracks = tracks if tracks else []
        self.lowered_tracks = [track.lower() if track else ""for track in tracks] if tracks else []
        self.used = [False] * len(self.tracks)
        self.lock = threading.Lock()

    def get_track_index(self, name, readonly: bool):
        """
        Finds the index of a track by its name.

        Args:
            name (str): The name of the track.

        Returns:
            int: The index of the track, or -1 if not found.
        """
        with self.lock:
            index = find_closest_string_index(name.lower(), self.lowered_tracks, self.used)
        if readonly:
            return index
        if index != -1:
            # Existing track found
            with self.lock:
                self.used[index] = True
            return index

        # Add new track
        with self.lock:
            index = len(self.tracks)
            self.used.append(True)
            self.tracks.append(name)
            self.lowered_tracks.append(name.lower())
        return index
    
    def get_folder_name(self):
        """
        Returns a sanitized folder name for the album.

        Returns:
            str: The sanitized folder name.
        """
        return fix_for_path(f"{self.artist} - {self.name}")
    
    def print(self):
        print(f'''
Album Information:
    Artist: {self.artist}
    Title: {self.name}
    Year: {self.year if self.year else UNKNOWN}
    Genre: {self.genre if self.genre else UNKNOWN}
    Tracks: {str(len(self.tracks)) if self.tracks else UNKNOWN}''')
        for i, track in enumerate(self.tracks):
            print(f"\t{i + 1}) {track if track else UNKNOWN}")
    
    def clear_temporary_data(self):
        with self.lock:
            self.used = [False] * len(self.tracks)
    
def _update_metadata(file_path, title, artist, album, track_number, year=None, genre=None, artwork=None):
    """
    Updates the metadata of an MP3 file.

    Args:
        file_path (str): Path to the MP3 file.
        title (str): Title of the track.
        artist (str): Artist name.
        album (str): Album name.
        track_number (int): Track number.
        genre (str, optional): Genre of the track.
        year (str, optional): Release year of the album.
        artwork (str, optional): Path to the album artwork image.
    """
    
    try:
        # Load the MP3 file
        audio = EasyID3(file_path)

        # Update metadata
        audio["title"] = title
        audio["artist"] = artist
        audio["album"] = album
        audio["tracknumber"] = str(track_number)
        if year:
            audio["date"] = year
        if genre:
            audio["genre"] = genre

        # Save changes
        audio.save()

        # Add artwork if it exists
        if artwork:
            # Get file extension type
            type = os.path.splitext(artwork)[1][1:].lower()
            if type == "jpg":
                type = "jpeg"

            if type not in ["jpeg", "png"]:
                print(f"{COLOR_ERROR}Unsupported artwork image format: {type}. Supported formats are: jpg, jpeg, png.{COLOR_RESET}")
                return

            # Load the ID3 tag and add the artwork
            audio = ID3(file_path)
            with open(artwork, "rb") as img:
                audio["APIC"] = APIC(
                    encoding=3,  # UTF-8
                    mime=f"image/{type}",  # MIME type of the image (e.g., "image/jpeg" or "image/png")
                    type=3,  # Front cover
                    desc="Cover",
                    data=img.read()  # Image data
                )

                # Save changes
            audio.save()
        
        # DEBUG: print values
    except Exception as e:
        print(f"{COLOR_ERROR}Failed to update metadata for \"{file_path}\". Error: {e}{COLOR_RESET}")

def update_metadata(file_path, album: Album, title: Title):
    """
    Updates the metadata of an audio file.

    Args:
        file_path (str): Path to the audio file.
        album (Album): Album object containing metadata.
    """
    _update_metadata(file_path, title.display_name, album.artist, album.name, title.track_index + 1, album.year, album.genre, album.artwork)

def parse_timestamps(description):
    """
    Parses timestamps and segment names from the video description.

    Args:
        description (str): The video description.

    Returns:
        list: A list of tuples containing (start_time, segment_name).
    """
    lines = description.splitlines()
    matches = list()
    # Regular expression to match timestamps in both formats:
    # 1. "0:00 Segment Name"
    # 2. "Segment Name 0:00"
    TIME_PATTERN = r"\[?(\d+:\d{2}(?::\d{2})?)\]?"
    NUMBER_PATTERN = r"(?:\d+[\.\)\s])?\s*"
    NAME_PATTERN = r"\W*(\w.*)"
    before_pattern = fr"^{NUMBER_PATTERN}{TIME_PATTERN}\s+{NAME_PATTERN}"
    after_pattern = fr"^{NUMBER_PATTERN}{NAME_PATTERN}\s+{TIME_PATTERN}"
    for line in lines:
        for match in re.findall(before_pattern, line):
            matches.append([match[0], match[1]])
        for match in re.findall(after_pattern, line):
            matches.append([match[1], match[0]])

    segments = []
    for time, name in matches:
        # Split the time string into components
        time_parts = list(map(int, time.split(":")))
        # Convert to total seconds (supports hours:minutes:seconds or minutes:seconds)
        total_seconds = time_parts[-1] # seconds
        total_seconds += time_parts[-2] * 60 # minutes
        if len(time_parts) >= 3: total_seconds += time_parts[-3] * 3600 # hours
        segments.append((total_seconds, name.strip()))
    return segments

def split_audio(file_path, segments, output_folder, format, album):
    """
    Splits the audio file into segments based on timestamps.

    Args:
        file_path (str): Path to the audio file.
        segments (list): List of tuples (start_time, segment_name).
        output_folder (str): Folder to save the split audio files.
    """
    lock = threading.Lock()
    def split_segment(i, start_time, name, audio):
        global g_audio_split_worker
        nonlocal lock, format, album, output_folder
        title = Title(name)
        title.fix_title(album, False)
        title_str = str(title)
        track_number = title.track_index + 1 if title.track_index != -1 else i + 1
        try:
            params = audio.getparams()
            frame_rate = params.framerate

            # Determine the start and end frames
            start_frame = int(start_time * frame_rate)
            end_frame = int(segments[i + 1][0] * frame_rate) if i + 1 < len(segments) else params.nframes

            # Set the position and read frames
            with lock:
                audio.setpos(start_frame)
                frames = audio.readframes(end_frame - start_frame)

            if not g_audio_split_worker.start_task(title_str, i):
                g_audio_split_worker.abort_task()
                add_download_result(RESULT_SKIPPED, title_str, "Segment was skipped", track_number)
                return
            g_audio_split_worker.increment(title_str)

            output_path = os.path.join(output_folder, f"{title_str}.wav")
            with wave.open(output_path, "wb") as segment:
                segment.setparams(params)
                segment.writeframes(frames)
            g_audio_split_worker.increment(title_str)

            final_file = convert_file(output_path, format)
            if not final_file:
                g_audio_split_worker.cancel_task(title_str)
                add_download_result(RESULT_ERROR, title_str, "Segment conversion failed", track_number)
                return
            g_audio_split_worker.increment(title_str)

            update_metadata(final_file, album, title)
            g_audio_split_worker.increment(title_str)
            add_download_result(RESULT_SUCCESS, title_str, final_file, track_number)
        except Exception as e:
            g_audio_split_worker.cancel_task(title_str)
            add_download_result(RESULT_ERROR, title_str, f"Segment processing failed: {e}", track_number)

    with wave.open(file_path, "rb") as audio:
        threads = []
        global g_audio_split_worker
        g_audio_split_worker = Worker(4, len(segments))
        for i, (start_time, name) in enumerate(segments):
            thread = threading.Thread(target=split_segment, args=(i, start_time, name, audio))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

def get_path_with_format(path, format):
    """
    Returns the path with the specified format.

    Args:
        path (str): The original file path.
        format (str): The desired output format (e.g., 'mp3', 'wav', 'mp4a').

    Returns:
        str: The path with the new format.
    """
    base, _ = os.path.splitext(path)
    if format.startswith('.'):
        format = format[1:]  # Remove leading dot if present
    return f"{base}.{format}"

def convert_file(path, format, trim_silence = TRIM_SILENCE):
    """
    Converts a file to the specified format.

    Args:
        path (str): Path to the file to convert.
        format (str): Desired output format (e.g., 'mp3', 'wav', 'mp4a').
        trim_silence (bool): Whether to trim silence at the start and end.

    Returns:
        str: Path to the converted file.
    """
    # If file already exists in the target format, increment the path
    target_path = get_path_with_format(path, format)
    # If the file is already in the desired format, return the original path
    if path == target_path:
        return path
    # Get the folder path and title from the original path
    base = os.path.dirname(path)
    title = os.path.splitext(os.path.basename(path))[0]
    while os.path.exists(target_path):
        title = increment_or_add_suffix(title)
        target_path = os.path.join(base, f"{title}.{format}")

    # Define the ffmpeg command for conversion
    command = [
        FFMPEG_PATH,
        "-i", path,  # Input file
        "-vn",  # No video
    ]

    if trim_silence:
        # Trim only edge silence (front/back) while keeping interior pauses.
        command += [
            "-af",
            f"silenceremove=start_periods=1:start_duration={TRIM_SILENCE_MIN_DURATION}:start_threshold={TRIM_SILENCE_THRESHOLD},areverse,silenceremove=start_periods=1:start_duration={TRIM_SILENCE_MIN_DURATION}:start_threshold={TRIM_SILENCE_THRESHOLD},areverse"
        ]

    # Add format-specific options
    if format == "mp3":
        command += ["-ar", "44100", "-ac", "2", "-b:a", "192k"]  # MP3 options
    elif format == "wav":
        command += ["-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2"]  # WAV options
    elif format == "mp4a":
        command += ["-c:a", "aac", "-b:a", "192k"]  # MP4A options
    else:
        raise ValueError(f"Unsupported format: {format}")

    # Add the target file path to the command
    command.append(target_path)

    # Run the ffmpeg command and suppress output
    try:
        with open(os.devnull, 'wb') as devnull:
            subprocess.run(command, check=True, stdout=devnull, stderr=devnull)
        os.remove(path)  # Optionally remove the original file after conversion
    except Exception as e:
        if trim_silence:
            # Some files can fail silence filters; retry conversion without trimming.
            retry_command = [
                FFMPEG_PATH,
                "-i", path,
                "-vn",
            ]

            if format == "mp3":
                retry_command += ["-ar", "44100", "-ac", "2", "-b:a", "192k"]
            elif format == "wav":
                retry_command += ["-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2"]
            elif format == "mp4a":
                retry_command += ["-c:a", "aac", "-b:a", "192k"]

            retry_command.append(target_path)
            try:
                with open(os.devnull, 'wb') as devnull:
                    subprocess.run(retry_command, check=True, stdout=devnull, stderr=devnull)
                os.remove(path)
                return target_path
            except Exception as retry_error:
                print(f"{COLOR_ERROR}Failed to convert {path} to {format}. Error: {retry_error}{COLOR_RESET}")
                return None

        print(f"{COLOR_ERROR}Failed to convert {path} to {format}. Error: {e}{COLOR_RESET}")
        return None

    return target_path

MUSICBRAINZ_DELAY = 1.1
MUSICBRAINZ_RETRIES = 3

def musicbrainz_delay():
    """
    Introduces a delay to respect MusicBrainz rate limiting.
    """
    time.sleep(MUSICBRAINZ_DELAY)

def setup_musicbrainz():
    """
    Sets up the MusicBrainz client with user agent and rate limiting.
    """
    musicbrainzngs.set_useragent(APP_NAME, APP_VERSION, APP_URL)
    musicbrainzngs.set_rate_limit(limit_or_interval=1.0, new_requests=1)
    musicbrainz_delay()

def safe_musicbrainz_request(request_func, *args, **kwargs):
    """Wrapper to safely make MusicBrainz requests with retry logic."""
    for attempt in range(MUSICBRAINZ_RETRIES):
        try:
            result = request_func(*args, **kwargs)
            return result
        except Exception as e:
            if attempt < MUSICBRAINZ_RETRIES - 1:
                # Exponential backoff with jitter
                delay = MUSICBRAINZ_DELAY * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
            else:
                raise e

def get_artist_info(artist_name):
    """
    Fetches artist information from MusicBrainz.

    Args:
        artist_name (str): The name of the artist.

    Returns:
        dict: Artist information including genres and tags.
    """
    def normalize_name(name):
        return re.sub(r"\s+", " ", name.strip().lower())

    def has_alias_match(artist, target_normalized):
        for alias in artist.get("alias-list", []):
            alias_name = alias.get("alias", "")
            if normalize_name(alias_name) == target_normalized:
                return True
        return False

    def to_score(value):
        try:
            return int(value)
        except Exception:
            return -1

    def similarity(a, b):
        return difflib.SequenceMatcher(None, a, b).ratio()

    def candidate_rank(artist, target_normalized):
        artist_name_normalized = normalize_name(artist.get("name", ""))
        alias_match = has_alias_match(artist, target_normalized)
        exact_name = artist_name_normalized == target_normalized
        starts_with = artist_name_normalized.startswith(target_normalized) or target_normalized.startswith(artist_name_normalized)
        contains = target_normalized in artist_name_normalized or artist_name_normalized in target_normalized
        return (
            1 if exact_name else 0,
            1 if alias_match else 0,
            1 if starts_with else 0,
            1 if contains else 0,
            similarity(artist_name_normalized, target_normalized),
            to_score(artist.get("ext:score", artist.get("score", -1)))
        )

    try:
        # Search for the artist by name. Use strict matching first, then broad search.
        strict_result = safe_musicbrainz_request(
            musicbrainzngs.search_artists,
            artist=artist_name,
            limit=100,
            strict=True
        )
        musicbrainz_delay()

        strict_list = strict_result.get("artist-list", []) if strict_result else []

        broad_result = safe_musicbrainz_request(
            musicbrainzngs.search_artists,
            artist=artist_name,
            limit=100
        )
        broad_list = broad_result.get("artist-list", []) if broad_result else []
        if not broad_list and not strict_list:
            return {"not found": "Artist not found"}
        musicbrainz_delay()

        # Prefer strict candidates when available.
        artist_list = strict_list if strict_list else broad_list
        target_normalized = normalize_name(artist_name)

        # Choose best candidate by exact/alias/prefix/contains/similarity/score.
        best_artist = max(artist_list, key=lambda a: candidate_rank(a, target_normalized))

        artist_id = best_artist["id"]

        # Fetch artist details including tags
        artist_details = safe_musicbrainz_request(musicbrainzngs.get_artist_by_id, artist_id, includes=["tags"])
        musicbrainz_delay()

        return artist_details["artist"]
    except Exception as e:
        return {"error": f"An error occurred while getting the artist info: {e}"}

def get_album_info(album_name, artist_info):
    """
    Fetches album information from MusicBrainz.

    Args:
        album_name (str): The name of the album.
        artist_name (str): The name of the artist.

    Returns:
        dict: Album information including release date, tracks, and more.
    """
    def get_list_item_with_highest_count(items, key):
        if not items:
            return None
        return max(items, key=lambda item: item.get(key, 0))

    def is_commentary_track(track_name):
        if not track_name:
            return False
        return re.search(r"\bcommentary\b", track_name, re.IGNORECASE) is not None

    def get_release_tracks(release_details):
        tracks = []
        for medium in release_details["release"].get("medium-list", []):
            for track in medium.get("track-list", []):
                if "title" in track:
                    tracks.append(track["title"])
                else:
                    tracks.append(track["recording"]["title"])
        return tracks

    valid_releases = []
    album_info = None

    try:
        # Search for all releases by name and artist to find the oldest
        result = safe_musicbrainz_request(musicbrainzngs.search_releases, release=album_name, artist=artist_info["name"], limit=100)
        if not result["release-list"]:
            return {"not found": "Album not found"}
    except Exception as e:
        return {"error": f"An error occurred while getting the album info: {e}"}
    
    musicbrainz_delay()

    try:
        if result["release-list"]:
            # Filter releases that match exactly and have dates
            valid_releases = []
            for release in result["release-list"]:
                if (release["title"].lower() == album_name.lower() and 
                    release["artist-credit"][0]["artist"]["name"].lower() == artist_info["name"].lower() and
                    release.get("date")):
                    valid_releases.append(release)
            
            if not valid_releases:
                # Fall back to first result if no exact matches with dates
                valid_releases = [result["release-list"][0]]
            
            # Find the release with the earliest date
            oldest_release = min(valid_releases, key=lambda r: int(r.get("date", "9999-99-99")[:4]))
            newest_release = max(valid_releases, key=lambda r: int(r.get("date", "0000-00-00")[:4]))

            genre = get_list_item_with_highest_count(newest_release.get("tag-list", []), "count") if "tag-list" in newest_release else None
            if genre:
                genre = genre.get("name").title()

            album_info = {
                "name": newest_release["title"],
                "artist": newest_release["artist-credit"][0]["artist"]["name"],
                "release_date": oldest_release.get("date", "Unknown"),
                "genre": genre,
                "tracks": [],
                "url": f"https://musicbrainz.org/release/{newest_release['id']}"
            }

            # Try to get genre information from the artist
            if not album_info["genre"]:
                try:
                    if "tag-list" in artist_info and artist_info["tag-list"]:
                        # Get the most popular tag as genre
                        tags = artist_info["tag-list"]
                        genre = get_list_item_with_highest_count(tags, "count")
                        if genre:
                            album_info["genre"] = genre.get("name").title()
                except Exception:
                    pass  # Keep genre as None if artist lookup fails
    except Exception as e:
        return {"error": f"An error occurred while building the album info: {e}"}

    musicbrainz_delay()

    try:
            if not valid_releases or album_info is None:
                return {"not found": "Album not found"}

            # Fetch track information from the best non-commentary release.
            candidate_releases = sorted(
                valid_releases,
                key=lambda r: r.get("track-count", r.get("medium-track-count", 0)),
                reverse=True
            )

            selected_release_id = None
            selected_tracks = None

            for candidate_release in candidate_releases:
                release_id = candidate_release["id"]
                release_details = safe_musicbrainz_request(musicbrainzngs.get_release_by_id, release_id, includes=["recordings"])
                musicbrainz_delay()

                candidate_tracks = get_release_tracks(release_details)
                if not candidate_tracks:
                    continue

                commentary_count = sum(1 for track in candidate_tracks if is_commentary_track(track))
                if commentary_count == len(candidate_tracks):
                    print(f"{COLOR_INFO}Skipping commentary release: {release_id}.{COLOR_RESET}")
                    continue

                selected_release_id = release_id
                selected_tracks = candidate_tracks
                break

            if not selected_tracks:
                # If all candidates look like commentary releases, fall back to the top candidate.
                fallback_release = candidate_releases[0]
                selected_release_id = fallback_release["id"]
                release_details = safe_musicbrainz_request(musicbrainzngs.get_release_by_id, selected_release_id, includes=["recordings"])
                selected_tracks = get_release_tracks(release_details)

            album_info["tracks"] = selected_tracks
            album_info["url"] = f"https://musicbrainz.org/release/{selected_release_id}"

            return album_info
    except Exception as e:
        return {"error": f"An error occurred while getting the album track info: {e}"}

def search_wikipedia_album_artwork(album_name, artist_name):
    """
    Searches Wikipedia for an album or related page and returns the direct link to its cover artwork image.
    
    Args:
        album_name (str): The name of the album.
        artist_name (str): The name of the artist.
    
    Returns:
        str or None: Direct URL to the album artwork image, or None if not found.
    """
    try:
        # Wikipedia API endpoint
        wiki_api_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        
        # Headers to avoid being blocked
        headers = {
            'User-Agent': 'AlbumWorks/1.0 (https://github.com/mtalyat/AlbumWorks; contact@example.com)'
        }
        
        # Try different search variations
        search_terms = [
            f"{album_name} ({artist_name} album)",
            f"{album_name} album {artist_name}",
            f"{album_name}",
            f"{album_name} (album)",
            f"{album_name} {artist_name}",
            f"{album_name} (video game)",
            f"{album_name} ({artist_name})",
            f"{artist_name} {album_name}"
        ]
        
        for search_term in search_terms:
            # Format the search term for URL
            formatted_term = search_term.replace(" ", "_")
            
            try:
                # Get page summary
                response = requests.get(f"{wiki_api_url}{formatted_term}", headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check if this looks like an album, soundtrack, or related page
                    description = data.get("description", "").lower()
                    extract = data.get("extract", "").lower()
                    
                    # Look for album, soundtrack, video game, or music-related keywords
                    relevant_keywords = ["album", "soundtrack", "video game", "music", "single", "ep", "ost"]
                    
                    if any(keyword in description or keyword in extract for keyword in relevant_keywords):
                        # Get the thumbnail image
                        if "thumbnail" in data and "source" in data["thumbnail"]:
                            thumbnail_url = data["thumbnail"]["source"]
                            
                            # Try to get the full resolution image
                            full_image_url = get_full_wikipedia_image(data.get("title", ""), thumbnail_url)
                            
                            return full_image_url or thumbnail_url
                        
            except requests.exceptions.RequestException:
                continue  # Try next search term
        
        # If no direct match found, try Wikipedia search API
        return search_wikipedia_with_search_api(album_name, artist_name)
        
    except Exception as e:
        print(f"{COLOR_ERROR}Error searching Wikipedia: {e}{COLOR_RESET}")
        return None

def get_full_wikipedia_image(page_title, thumbnail_url):
    """
    Attempts to get the full resolution version of a Wikipedia image.
    
    Args:
        page_title (str): The Wikipedia page title.
        thumbnail_url (str): The thumbnail URL.
    
    Returns:
        str or None: Full resolution image URL, or None if not found.
    """
    try:
        # Headers for requests
        headers = {
            'User-Agent': 'AlbumWorks/1.0 (https://github.com/mtalyat/AlbumWorks; contact@example.com)'
        }
        
        # Extract the filename from the thumbnail URL
        if "/thumb/" in thumbnail_url:
            # Extract the original filename
            parts = thumbnail_url.split("/thumb/")
            if len(parts) > 1:
                # Get the part after /thumb/
                after_thumb = parts[1]
                # Split by / and take the first part (original filename)
                filename = after_thumb.split("/")[0]
                
                # Construct the full resolution URL
                full_url = f"https://upload.wikimedia.org/wikipedia/en/{filename}"
                
                # Test if the full resolution image exists
                response = requests.head(full_url, headers=headers)
                if response.status_code == 200:
                    return full_url
                
                # Try commons if en doesn't work
                full_url = f"https://upload.wikimedia.org/wikipedia/commons/{filename}"
                response = requests.head(full_url, headers=headers)
                if response.status_code == 200:
                    return full_url
        
        return None
        
    except Exception:
        return None

def search_wikipedia_with_search_api(album_name, artist_name):
    """
    Uses Wikipedia's search API as a fallback method.
    
    Args:
        album_name (str): The name of the album.
        artist_name (str): The name of the artist.
    
    Returns:
        str or None: Image URL if found, None otherwise.
    """
    try:
        search_api_url = "https://en.wikipedia.org/api/rest_v1/page/search"
        
        # Headers to avoid being blocked
        headers = {
            'User-Agent': 'AlbumWorks/1.0 (https://github.com/mtalyat/AlbumWorks; contact@example.com)'
        }
        
        # Search for the album
        search_query = f"{album_name} {artist_name} album"
        params = {
            "q": search_query,
            "limit": 5
        }
        
        response = requests.get(search_api_url, params=params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            if "pages" in data:
                for page in data["pages"]:
                    # Check if this looks like an album page
                    if "album" in page.get("description", "").lower():
                        # Get the page summary to find the image
                        page_title = page["title"].replace(" ", "_")
                        summary_response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title}", headers=headers)
                        
                        if summary_response.status_code == 200:
                            summary_data = summary_response.json()
                            
                            if "thumbnail" in summary_data and "source" in summary_data["thumbnail"]:
                                thumbnail_url = summary_data["thumbnail"]["source"]
                                full_image_url = get_full_wikipedia_image(page_title, thumbnail_url)
                                
                                return full_image_url or thumbnail_url
        
        return None
        
    except Exception:
        return None

def download_youtube_video(url, output_folder, format, album, i = 0, limit = None, ignore_segments = False, title_hint = None):
    """
    Downloads a YouTube video, converts it to WAV, splits it into segments, and saves it in the specified folder.

    Args:
        url (str): The URL of the YouTube video.
        output_folder (str): The folder where the video and its segments will be saved.
    """
    global g_download_worker
    title_str = None
    track_number = i + 1 if i is not None and i >= 0 else None
    try:
        if title_hint:
            hinted_title = Title(title_hint)
            hinted_path = os.path.join(output_folder, f'{hinted_title.file_name}.{format}')
            if album:
                for idx, track_name in enumerate(album.tracks):
                    if track_name and track_name.lower() == hinted_title.display_name.lower():
                        track_number = idx + 1
                        break
            if os.path.exists(hinted_path):
                g_download_worker.abort_task()
                add_download_result(RESULT_SKIPPED, hinted_title.display_name, "File already exists", track_number)
                return None

        normalized_url = get_video_url(url)
        if not normalized_url:
            print(f"{COLOR_ERROR}Skipping invalid video URL: {url}{COLOR_RESET}")
            add_download_result(RESULT_SKIPPED, url, "Invalid video URL", track_number)
            g_download_worker.abort_task()
            return None

        # Create a YouTube object
        yt = create_youtube_client(normalized_url)

        # Parse timestamps from the description
        description = yt.description
        if ignore_segments: # do not parse segments when in a playlist
            segments = []
        else:
            segments = parse_timestamps(description)

        # Limit the number of segments if specified
        if limit:
            segments = segments[:limit]

        # Get video title and description
        title = yt.title

        # Print index and title, if no segments found
        title = Title(yt.title)
        if not segments: # this video is the song
            # If this is the only video being downloaded AND there is only one song in the album, use the album track list to fix the title
            if g_download_worker.task_count == 1 and album and len(album.tracks) == 1:
                title.set_title(album.tracks[0], 0)
            else:
                title.fix_title(album, False)
        else: # this video has segments (each segment is a song)
            title.fix_title(None, True)
        title_str = str(title)
        if title.track_index != -1:
            track_number = title.track_index + 1

        if not g_download_worker.start_task(title_str, i):
            g_download_worker.abort_task()
            add_download_result(RESULT_SKIPPED, title_str, "Skipped by priority", track_number)
            return
        
        # If song is already downloaded, skip
        if os.path.exists(os.path.join(output_folder, f'{title.file_name}.{format}')):
            g_download_worker.cancel_task(title_str)
            add_download_result(RESULT_SKIPPED, title_str, "File already exists", track_number)
            return

        g_download_worker.increment(title_str)

        # Get the audio stream
        audio_stream = yt.streams.filter(only_audio=True).first()
        if not audio_stream:
            print(f"{COLOR_ERROR}No audio stream available for video: {yt.title}{COLOR_RESET}")
            g_download_worker.cancel_task(title_str)
            add_download_result(RESULT_ERROR, title_str, "No audio stream available", track_number)
            return

        # Download the audio to the output folder
        downloaded_file = audio_stream.download(output_path=output_folder)
        if not downloaded_file or not os.path.exists(downloaded_file):
            print(f"{COLOR_ERROR}Failed to download audio for video: {yt.title}{COLOR_RESET}")
            g_download_worker.cancel_task(title_str)
            add_download_result(RESULT_ERROR, title_str, "Audio download failed", track_number)
            return None

        g_download_worker.increment(title_str)

        # Rename to use the safe title
        file_name = TEMP_NAME if segments else title.file_name
        file_name = os.path.join(output_folder, f"{file_name}.mp4")

        # Normalize temporary name so conversion always targets a known path.
        if os.path.abspath(downloaded_file) != os.path.abspath(file_name):
            if os.path.exists(file_name):
                os.remove(file_name)
            os.replace(downloaded_file, file_name)
        downloaded_file = file_name

        g_download_worker.increment(title_str)
        
        # if no segments are found, convert to target format and return (this is a single song video)
        if not segments:
            final_file = convert_file(downloaded_file, format)
            if not final_file:
                print(f"{COLOR_ERROR}Failed to convert audio for video: {yt.title}{COLOR_RESET}")
                g_download_worker.cancel_task(title_str)
                add_download_result(RESULT_ERROR, title_str, "Audio conversion failed", track_number)
                return None
            update_metadata(final_file, album, title)
            g_download_worker.increment(title_str)
            add_download_result(RESULT_SUCCESS, title_str, final_file, track_number)
            return final_file

        # There are segments, so proceed to split the audio into multiple songs

        # Convert the downloaded file to WAV format using ffmpeg so it can be edited
        wav_file = convert_file(downloaded_file, "wav", trim_silence=False)
        if not wav_file:
            print(f"{COLOR_ERROR}Failed to prepare audio segments for video: {yt.title}{COLOR_RESET}")
            g_download_worker.cancel_task(title_str)
            add_download_result(RESULT_ERROR, title_str, "Failed to prepare split segments", track_number)
            return None

        g_download_worker.increment(title_str)

        # Create a folder for the video's segments
        os.makedirs(output_folder, exist_ok=True)

        # Split the audio into segments
        split_audio(wav_file, segments, output_folder, format, album)

        # Delete the temporary WAV file
        os.remove(wav_file)

        return output_folder
    except Exception as e:
        print(f"{COLOR_ERROR}An error occurred while processing video: {url}. Error: {e}{COLOR_RESET}")
        g_download_worker.cancel_task(title_str)
        add_download_result(RESULT_ERROR, title_str if title_str else url, str(e), track_number)
        return None

def download_youtube_playlist(playlist_url, output_folder, format, album, song_limit = None):
    """
    Downloads all videos in a YouTube playlist.

    Args:
        playlist_url (str): The URL of the YouTube playlist.
    """
    try:        
        playlist_url = get_playlist_url(playlist_url) or playlist_url

        # Create a Playlist object
        playlist = Playlist(playlist_url)

        urls = playlist.video_urls
        if not urls:
            try:
                urls = [video.watch_url for video in playlist.videos if hasattr(video, "watch_url")]
            except Exception:
                pass
        if not urls:
            urls = extract_playlist_video_urls(playlist_url)

        # Normalize to valid watch URLs and drop non-video links (e.g. search URLs).
        normalized_urls = []
        seen_urls = set()
        for candidate_url in urls:
            normalized_url = get_video_url(candidate_url)
            if not normalized_url or normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)
            normalized_urls.append(normalized_url)
        urls = normalized_urls

        if song_limit:
            urls = urls[:song_limit]

        if not urls:
            print(f"{COLOR_ERROR}No videos found in playlist: {playlist_url}{COLOR_RESET}")
            return None

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Iterate through all videos in the playlist
        global g_download_worker
        g_download_worker = Worker(4, len(urls))
        threads = []
        results = [None] * len(urls)

        def download_video_at_index(index, video_url, track_hint):
            results[index] = download_youtube_video(video_url, output_folder, format, album, index, None, True, track_hint)

        for i, video_url in enumerate(urls):
            track_hint = album.tracks[i] if album and i < len(album.tracks) else None

            # Pre-check with known track names to skip files before making any network request.
            if track_hint:
                hinted_title = Title(track_hint)
                hinted_path = os.path.join(output_folder, f'{hinted_title.file_name}.{format}')
                if os.path.exists(hinted_path):
                    g_download_worker.abort_task()
                    add_download_result(RESULT_SKIPPED, hinted_title.display_name, "File already exists", i + 1)
                    continue

            thread = threading.Thread(target=download_video_at_index, args=(i, video_url, track_hint))
            threads.append(thread)
            thread.start()
            # Avoid bursting requests too quickly, which can trigger bot detection.
            if i < len(urls) - 1 and YOUTUBE_REQUEST_SPACING_SECONDS > 0:
                time.sleep(YOUTUBE_REQUEST_SPACING_SECONDS)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        return output_folder
    except Exception as e:
        print(f"{COLOR_ERROR}An error occurred while processing the playlist: {playlist_url}. Error: {e}{COLOR_RESET}")
        return None

def download_youtube_thumbnail(url, output_folder, name):
    """
    Downloads the thumbnail of a YouTube video or the first video in a playlist.

    Args:
        url (str): The URL of the YouTube video or playlist.
        output_folder (str): The folder where the thumbnail will be saved.

    Returns:
        str: Path to the downloaded thumbnail, or None if an error occurs.
    """
    try:
        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Check if the URL contains playlist information
        playlist_url = get_playlist_url(url)
        if playlist_url is not None:
            # Handle playlist: use the first video's thumbnail
            playlist = Playlist(playlist_url)
            playlist_video_urls = playlist.video_urls
            if not playlist_video_urls:
                try:
                    playlist_video_urls = [video.watch_url for video in playlist.videos if hasattr(video, "watch_url")]
                except Exception:
                    playlist_video_urls = []
            if not playlist_video_urls:
                playlist_video_urls = extract_playlist_video_urls(playlist_url)
            if not playlist_video_urls:
                return None
            video_url = playlist_video_urls[0]  # Use the first video in the playlist
        else:
            # Handle single video
            video_url = url

        # Create a YouTube object for the selected video
        yt = create_youtube_client(video_url)

        # Get the thumbnail URL
        thumbnail_url = yt.thumbnail_url

        # Define the output file path
        thumbnail_path = os.path.join(output_folder, f"{name}.jpg")

        # Download the thumbnail
        response = requests.get(thumbnail_url, stream=True)
        if response.status_code == 200:
            with open(thumbnail_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            return thumbnail_path
        else:
            print(f"{COLOR_ERROR}Failed to download thumbnail. HTTP Status Code: {response.status_code}{COLOR_RESET}")
            return None

    except Exception as e:
        print(f"{COLOR_ERROR}An error occurred while downloading the thumbnail: {e}{COLOR_RESET}")
        return None

def download_image(url, output_folder, name):
    """
    Downloads an image from a URL and saves it to the specified folder.

    Args:
        url (str): The URL of the image to download.
        output_folder (str): The folder where the image will be saved.
        name (str): The name of the image file (without extension).

    Returns:
        str: Path to the downloaded image, or None if an error occurs.
    """
    try:
        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Define the output file path
        image_path = os.path.join(output_folder, f"{name}.jpg")

        # Set headers to mimic a real browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        # Download the image
        response = requests.get(url, stream=True, headers=headers)
        if response.status_code == 200:
            with open(image_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            return image_path
        else:
            print(f"{COLOR_ERROR}Failed to download image. HTTP Status Code: {response.status_code}{COLOR_RESET}")
            return None

    except Exception as e:
        print(f"{COLOR_ERROR}An error occurred while downloading the image: {e}{COLOR_RESET}")
        return None

def open_directory(path):
    """
    Opens the given directory in the system file explorer.

    Args:
        path (str): The path to the directory to open.
    """
    # Ensure the path exists
    if os.path.exists(path):
        os.startfile(path)  # Opens the directory in Windows Explorer
    else:
        print(f"{COLOR_ERROR}The directory does not exist: {path}{COLOR_RESET}")

def print_line():
    print("-" * os.get_terminal_size().columns)

def get_playlist_url(url):
    """
    Returns a normalized YouTube playlist URL when one is present, otherwise None.
    """
    try:
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        list_ids = query.get("list", [])
        if list_ids and len(list_ids[0]) > 0:
            return f"https://www.youtube.com/playlist?list={list_ids[0]}"
    except Exception:
        pass
    return None

def get_video_url(url):
    """
    Returns a normalized YouTube watch URL when a valid 11-character video ID
    can be extracted from the input URL, otherwise None.
    """
    try:
        parsed = urlparse(url)

        # Standard watch URL: ?v=<id>
        query = parse_qs(parsed.query)
        if "v" in query and len(query["v"]) > 0:
            candidate = query["v"][0]
            if re.fullmatch(r"[0-9A-Za-z_-]{11}", candidate):
                return f"https://www.youtube.com/watch?v={candidate}"

        # Short URL: youtu.be/<id>
        path = parsed.path.strip("/")
        if parsed.netloc.lower().endswith("youtu.be") and len(path) > 0:
            candidate = path.split("/")[0]
            if re.fullmatch(r"[0-9A-Za-z_-]{11}", candidate):
                return f"https://www.youtube.com/watch?v={candidate}"

        # Generic fallback: extract any embedded 11-char video id from URL
        match = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})(?:[?&/]|$)", url)
        if match:
            return f"https://www.youtube.com/watch?v={match.group(1)}"
    except Exception:
        pass

    return None

def extract_playlist_video_urls(playlist_url):
    """
    Extracts video watch URLs from a playlist page as a fallback when pytubefix
    fails to populate Playlist.video_urls (common with some OLAK playlists).
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(playlist_url, headers=headers, timeout=20)
        if response.status_code != 200:
            return []

        video_ids = re.findall(r'"videoId":"([A-Za-z0-9_-]{11})"', response.text)

        unique_ids = []
        seen = set()
        for video_id in video_ids:
            if video_id in seen:
                continue
            seen.add(video_id)
            unique_ids.append(video_id)

        return [f"https://www.youtube.com/watch?v={video_id}" for video_id in unique_ids]
    except Exception:
        return []

def _is_bot_detection_error(error):
    text = str(error).lower()
    return "detected as a bot" in text or "po_token" in text

def _get_po_token_verifier_from_env():
    visitor_data = os.getenv(YT_VISITOR_DATA_ENV)
    po_token = os.getenv(YT_PO_TOKEN_ENV)
    if not visitor_data or not po_token:
        return None
    return lambda _: (visitor_data, po_token)

def create_youtube_client(url):
    """
    Creates a resilient pytubefix YouTube client, retrying with alternate
    clients and optional PO token settings when bot detection is triggered.
    """
    bot_error = None

    try:
        return YouTube(url)
    except Exception as e:
        if not _is_bot_detection_error(e):
            raise
        bot_error = e

    po_token_verifier = _get_po_token_verifier_from_env()
    if po_token_verifier is not None:
        try:
            return YouTube(url, use_po_token=True, po_token_verifier=po_token_verifier)
        except Exception as e:
            if not _is_bot_detection_error(e):
                raise
            bot_error = e

    for client in ["TV", "WEB", "MWEB"]:
        try:
            return YouTube(url, client=client)
        except Exception as e:
            if not _is_bot_detection_error(e):
                raise
            bot_error = e

    print(
        f"{COLOR_INFO}YouTube bot detection is blocking this request. "
        f"Set {YT_VISITOR_DATA_ENV} and {YT_PO_TOKEN_ENV} environment variables to enable PO token fallback.{COLOR_RESET}"
    )

    raise bot_error if bot_error else RuntimeError("Failed to create YouTube client")

def main():
    print("Welcome to the AlbumWorks downloader!")
    print("You can download individual videos or entire playlists.")
    print()
    print("Setup:")

    # Configure MusicBrainz before making any requests.
    setup_musicbrainz()
    
    # Prompt for output format
    formats_list = ", ".join(FORMATS)
    format = input(f"Enter the desired output format ({formats_list}): ").strip().lower()
    if len(format) == 0:
        print(f"{COLOR_INFO}Defaulting to {FORMATS[0]}.{COLOR_RESET}")
        format = FORMATS[0]
    if format not in FORMATS:
        print(f"{COLOR_ERROR}Invalid format. Supported formats are: {formats_list}{COLOR_RESET}")
        return
    
    # Prompt for output folder
    base_output_folder = input(f"Enter the output folder (defaults to {OUTPUT_PATH}): ").strip()
    if len(base_output_folder) == 0:
        base_output_folder = OUTPUT_PATH
    if not os.path.exists(base_output_folder):
        print(f"{COLOR_ERROR}Output folder does not exist.{COLOR_RESET}")
        return
    print(f"{COLOR_INFO}Output folder set to: {base_output_folder}.{COLOR_RESET}")

    print()

    album_artist = ""
    album_name = ""
    os.system(f'title AlbumWorks')

    album_info_cache: list[Album] = list()
    artist_info_cache: list[tuple[str, dict]] = list()
    ALBUM_INFO_CACHE_SIZE = 10
    ARTIST_INFO_CACHE_SIZE = 20

    def get_cached_artist_info(artist_name):
        key = artist_name.lower()
        for i, (cached_key, artist_info) in enumerate(artist_info_cache):
            if cached_key == key:
                # Move to front of cache (most recently used)
                artist_info_cache.pop(i)
                artist_info_cache.insert(0, (cached_key, artist_info))
                return artist_info
        return None

    def cache_artist_info(artist_name, artist_info):
        key = artist_name.lower()

        # Remove existing entry for this key before inserting at the front
        for i, (cached_key, _) in enumerate(artist_info_cache):
            if cached_key == key:
                artist_info_cache.pop(i)
                break

        artist_info_cache.insert(0, (key, artist_info))
        if len(artist_info_cache) > ARTIST_INFO_CACHE_SIZE:
            artist_info_cache.pop()

    def parse_artist_names(artist_text):
        return [name.strip() for name in artist_text.split(";") if name.strip()]

    def normalize_artist_name(name):
        return re.sub(r"\s+", " ", (name or "").strip().lower())

    def combine_artist_names(primary_artist, additional_artists):
        names = []
        seen = set()
        for name in [primary_artist] + list(additional_artists):
            lowered = name.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            names.append(name)
        return ", ".join(names)

    def resolve_artist_canonical_name(artist_name):
        cached_info = get_cached_artist_info(artist_name)
        if cached_info is not None:
            if normalize_artist_name(cached_info.get("name", "")) == normalize_artist_name(artist_name):
                return cached_info.get("name", artist_name), cached_info

        info = get_artist_info(artist_name)
        if "error" in info or "not found" in info:
            return artist_name, {"name": artist_name}

        cache_artist_info(artist_name, info)
        canonical_name = info.get("name", artist_name)
        cache_artist_info(canonical_name, info)
        return canonical_name, info

    def prompt_tracks_input(existing_tracks = None):
        tracks = list(existing_tracks) if existing_tracks else []
        while True:
            track_input = input("Enter the index for track to edit, or the name of a new track to add (or leave blank to finish): ").strip()
            if len(track_input) == 0:
                break
            if track_input.isdigit():
                track_index = int(track_input) - 1
                while len(tracks) <= track_index:
                    tracks.append("")
                track_name = input(f"Enter the name for track {track_index + 1}: ").strip()
                tracks[track_index] = track_name
            else:
                tracks.append(track_input)
        return tracks

    def prompt_album_edit_or_url(album):
        while True:
            entry = input("Enter album attribute to edit, or YouTube URL to continue: ").strip()
            if len(entry) == 0:
                return None

            entry_parts = entry.split()
            url_candidate = entry_parts[0]
            limit_value = None

            if get_playlist_url(url_candidate) is not None or "youtu" in url_candidate.lower():
                if len(entry_parts) > 1:
                    try:
                        limit_value = int(entry_parts[1])
                    except ValueError:
                        print(f"{COLOR_ERROR}Invalid limit value: {entry_parts[1]}{COLOR_RESET}")
                        continue
                return (url_candidate, limit_value)

            choice = entry.lower()

            if choice == "artist":
                album.artist = input("Enter the new artist name: ").strip()
            elif choice in ["title", "name"]:
                album.name = input("Enter the new album title: ").strip()
            elif choice == "year":
                album.year = input("Enter the new album year (leave blank for unknown): ").strip()
                if len(album.year) == 0:
                    album.year = None
            elif choice == "genre":
                album.genre = input("Enter the new album genre (leave blank for unknown): ").strip()
                if len(album.genre) == 0:
                    album.genre = None
            elif choice == "tracks":
                album.tracks = prompt_tracks_input(album.tracks)
                album.lowered_tracks = [track.lower() if track else "" for track in album.tracks]
                album.clear_temporary_data()
            else:
                print(f"{COLOR_ERROR}Input must be a supported attribute or a YouTube URL.{COLOR_RESET}")
                continue

            print(f"{COLOR_INFO}Updated album information:{COLOR_RESET}")
            album.print()
    
    while True:
        print_line()
        output_folder = base_output_folder

        # Prompt for album information
        album_artist_input = input("Enter the album artist: ").strip()
        if len(album_artist_input) > 0:
            album_artist = album_artist_input
        elif len(album_artist) > 0:
            print(f"{COLOR_INFO}Album artist set to previous value: {album_artist}.{COLOR_RESET}")
        else:
            print(f"{COLOR_ERROR}No album artist given.{COLOR_RESET}")
            continue

        artist_names = parse_artist_names(album_artist)
        if not artist_names:
            print(f"{COLOR_ERROR}No album artist given.{COLOR_RESET}")
            continue

        search_artist = artist_names[0]
        additional_artists = artist_names[1:]

        artist_info = get_cached_artist_info(search_artist)
        if artist_info is not None:
            if normalize_artist_name(artist_info.get("name", "")) != normalize_artist_name(search_artist):
                artist_info = None
        if artist_info is not None:
            print(f"{COLOR_INFO}Using cached artist information.{COLOR_RESET}")
            album_name_input = input("Enter the album name: ").strip()
        else:
            # Get the artist info from MusicBrainz in the background while user types out the album name
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(get_artist_info, search_artist)

                album_name_input = input("Enter the album name: ").strip()

                artist_info = future.result()

            if "error" in artist_info or "not found" in artist_info:
                # Fall back to typed artist name so album lookup can still proceed.
                artist_info = {"name": search_artist}
            else:
                # Cache both the entered and canonical artist names
                cache_artist_info(search_artist, artist_info)
                canonical_artist_name = artist_info.get("name", search_artist)
                cache_artist_info(canonical_artist_name, artist_info)

        canonical_additional_artists = []
        for additional_artist in additional_artists:
            canonical_name, _ = resolve_artist_canonical_name(additional_artist)
            canonical_additional_artists.append(canonical_name)

        album_artist = combine_artist_names(artist_info.get("name", search_artist), canonical_additional_artists)
        if len(album_name_input) > 0:
            album_name = album_name_input
        elif len(album_name) > 0:
            print(f"{COLOR_INFO}Album name set to previous value: {album_name}.{COLOR_RESET}")
        else:
            print(f"{COLOR_ERROR}No album name given.{COLOR_RESET}")
            continue

        album = None
        for album_info in album_info_cache:
            if album_info.name.lower() == album_name.lower() and album_info.artist.lower() == album_artist.lower():
                album = album_info
                # Clear the album's temporary data
                album.clear_temporary_data()
        if album is not None:
            print(f"{COLOR_INFO}Using cached album information.{COLOR_RESET}")
            # Move to front of cache
            album_info_cache.remove(album)
            album_info_cache.insert(0, album)
        else:
            print(f"{COLOR_INFO}Retrieving album information...{COLOR_RESET}")
            album_info = get_album_info(album_name, artist_info)
            album_year = None
            album_tracks = None
            manual_entry = album_info is None or "not found" in album_info or "error" in album_info
            if manual_entry:
                print(f"{COLOR_INFO}Album not found.{COLOR_RESET}")
                if album_info is not None and "error" in album_info:
                    print(album_info["error"])
                print("Please enter the rest of the album info manually.")
                album_year = input("Enter the album year: ").strip()
                if len(album_year) == 0: album_year = None
                album_genre = input("Enter the album genre: ").strip()
                if len(album_genre) == 0: album_genre = None
                album_tracks = prompt_tracks_input([])
            else:
                album_name = album_info["name"]
                album_artist = combine_artist_names(album_info["artist"], canonical_additional_artists)
                album_year = album_info["release_date"][:4]
                album_genre = album_info["genre"]
                album_tracks = album_info["tracks"]
            album = Album(album_name, album_artist, album_year, album_genre, album_tracks, None)
            # Add to cache
            album_info_cache.insert(0, album)
            if len(album_info_cache) > ALBUM_INFO_CACHE_SIZE:
                album_info_cache.pop()

        # Print album information
        album.print()
        edit_or_url = prompt_album_edit_or_url(album)
        if edit_or_url is None:
            print(f"{COLOR_INFO}Input cancelled.{COLOR_RESET}")
            continue

        url, limit = edit_or_url

        # Keep local album values aligned with any manual edits.
        album_name = album.name
        album_artist = album.artist
        print()

        g_result_display.reset()

        # Search for the album artwork
        print(f"{COLOR_INFO}Searching for album artwork...{COLOR_RESET}")

        wiki_artwork_url = search_wikipedia_album_artwork(album_name, album_artist)
        if wiki_artwork_url:
            print(f"{COLOR_INFO}Found album artwork on Wikipedia: {wiki_artwork_url}{COLOR_RESET}")
            album_artwork = wiki_artwork_url
            album_artwork_type = ARTWORK_TYPE_LINK
        else:
            print(f"{COLOR_INFO}No artwork found on Wikipedia.{COLOR_RESET}")
            album_artwork = input("Enter the path to the album artwork, or nothing/\"thumbnail\" to use the YouTube thumbnail: ").strip()

            # Prompt for artwork, if user wants to override
            if len(album_artwork) == 0 or album_artwork.lower() == "thumbnail":
                album_artwork = download_youtube_thumbnail(url, OUTPUT_PATH, THUMBNAIL_NAME)
                album_artwork_type = ARTWORK_TYPE_THUMBNAIL
            elif album_artwork.lower() == "thumbnail":
                album_artwork_type = ARTWORK_TYPE_THUMBNAIL
            elif os.path.exists(album_artwork):
                album_artwork_type = ARTWORK_TYPE_FILE
            else:
                album_artwork_type = ARTWORK_TYPE_LINK

        # Make sure the output path exists
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        
        if album_artwork_type == ARTWORK_TYPE_LINK:
            # Download the artwork from the link
            album_artwork_temporary = True
            album_artwork = download_image(album_artwork, OUTPUT_PATH, THUMBNAIL_NAME)
        elif album_artwork_type == ARTWORK_TYPE_FILE:
            # Use the existing file
            album_artwork_temporary = False
            if not os.path.exists(album_artwork):
                print(f"{COLOR_ERROR}Artwork file does not exist: {album_artwork}{COLOR_RESET}")
                continue
        elif album_artwork_type == ARTWORK_TYPE_THUMBNAIL:
            # Use the YouTube thumbnail
            album_artwork_temporary = True
            album_artwork = download_youtube_thumbnail(url, OUTPUT_PATH, THUMBNAIL_NAME)
        else:
            print(f"{COLOR_ERROR}Invalid artwork type.{COLOR_RESET}")
            continue

        # Set the album artwork in the album object
        album.artwork = album_artwork

        album_path = os.path.join(output_folder, album.get_folder_name())
        playlist_url = get_playlist_url(url)
        if playlist_url is not None:
            # Process as a playlist
            print(f"{COLOR_INFO}Processing playlist...{COLOR_RESET}")
            output_folder = download_youtube_playlist(playlist_url, album_path, format, album, limit)
        elif "youtu" in url:
            # Process as a single video
            print(f"{COLOR_INFO}Processing video...{COLOR_RESET}")
            global g_download_worker
            g_download_worker = Worker(4, 1)
            single_track_hint = album.tracks[0] if album and len(album.tracks) == 1 else None
            output_folder = download_youtube_video(url, album_path, format, album, limit=limit, title_hint=single_track_hint)
            # Get the output folder from the returned file path
            if output_folder is not None and os.path.isfile(output_folder):
                output_folder = os.path.dirname(output_folder)
        else:
            print(f"{COLOR_ERROR}Invalid URL. Please provide a valid YouTube video or playlist URL.{COLOR_RESET}")
            continue

        g_result_display.print_results()
        g_result_display.print_summary()
        
        # Delete thumbnail if it was downloaded
        if album_artwork_temporary and os.path.exists(album_artwork):
            try:
                os.remove(album_artwork)
            except Exception as e:
                print(f"{COLOR_ERROR}Failed to delete thumbnail: {e}{COLOR_RESET}")
        
        # Done

if __name__ == "__main__":
    main()