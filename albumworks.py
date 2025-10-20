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

FFMPEG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "ffmpeg", "bin", "ffmpeg.exe")
OUTPUT_PATH = os.path.join(os.path.expanduser("~"), "Music")
FORMATS = ["mp3", "wav", "mp4a"]
TEMP_NAME = "temp"
THUMBNAIL_NAME = "thumbnail"
UNKNOWN = "Unknown"

ARTWORK_TYPES = ['file', 'url', 'thumbnail']
ARTWORK_TYPE_FILE = 0 # file on disk
ARTWORK_TYPE_LINK = 1 # link to an online image
ARTWORK_TYPE_THUMBNAIL = 2 # YouTube thumbnail

COLOR_ERROR = "\033[91m"
COLOR_SUCCESS = "\033[92m"
COLOR_INFO = "\033[90m"
COLOR_RESET = "\033[0m"

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
        quote_pattern = r'"(.+?)"'
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
            if string.startswith(target) and not used[i]:
                return i
        # try the opposite
        for i, string in enumerate(string_list):
            if string in target and not used[i]:
                return i
        # try matching closest string using difflib
        matches = difflib.get_close_matches(target, string_list, n=1, cutoff=0.8)
        if matches:
            closest_match = matches[0]
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
        self.lowered_tracks = [track.lower() for track in tracks] if tracks else []
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
        params = audio.getparams()
        frame_rate = params.framerate
        
        # Determine the start and end frames
        start_frame = int(start_time * frame_rate)
        end_frame = int(segments[i + 1][0] * frame_rate) if i + 1 < len(segments) else params.nframes

        # Set the position and read frames
        with lock:
            audio.setpos(start_frame)
            frames = audio.readframes(end_frame - start_frame)

        # Save the segment with the given name
        title = Title(name)
        title.fix_title(album, False)
        title_str = str(title)

        if not g_audio_split_worker.start_task(title_str, i):
            g_audio_split_worker.abort_task()
            return
        g_audio_split_worker.increment(title_str)

        output_path = os.path.join(output_folder, f"{title_str}.wav")
        with wave.open(output_path, "wb") as segment:
            segment.setparams(params)
            segment.writeframes(frames)
        g_audio_split_worker.increment(title_str)
        final_file = convert_file(output_path, format)
        g_audio_split_worker.increment(title_str)
        update_metadata(final_file, album, title)
        g_audio_split_worker.increment(title_str)

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

def convert_file(path, format):
    """
    Converts a file to the specified format.

    Args:
        path (str): Path to the file to convert.
        format (str): Desired output format (e.g., 'mp3', 'wav', 'mp4a').

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
        print(f"{COLOR_ERROR}Failed to convert {path} to {format}. Error: {e}{COLOR_RESET}")
        return None

    return target_path

def get_album_info(album_name, artist_name):
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

    # Set up MusicBrainz client
    musicbrainzngs.set_useragent("AlbumWorks", "1.0", "https://example.com")

    try:
        # Search for all releases by name and artist to find the oldest
        result = musicbrainzngs.search_releases(release=album_name, artist=artist_name, limit=100)
        if not result["release-list"]:
            return {"not found": "Album not found"}
    except Exception as e:
        return {"error": f"An error occurred while getting the album info: {e}"}
    
    try:
        if result["release-list"]:
            # Filter releases that match exactly and have dates
            valid_releases = []
            for release in result["release-list"]:
                if (release["title"].lower() == album_name.lower() and 
                    release["artist-credit"][0]["artist"]["name"].lower() == artist_name.lower() and
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
                    artist_id = newest_release["artist-credit"][0]["artist"]["id"]
                    artist_details = musicbrainzngs.get_artist_by_id(artist_id, includes=["tags"])
                    if "tag-list" in artist_details["artist"] and artist_details["artist"]["tag-list"]:
                        # Get the most popular tag as genre
                        tags = artist_details["artist"]["tag-list"]
                        genre = get_list_item_with_highest_count(tags, "count")
                        if genre:
                            album_info["genre"] = genre.get("name").title()
                except Exception:
                    pass  # Keep genre as None if artist lookup fails
    except Exception as e:
        return {"error": f"An error occurred while building the album info: {e}"}
    
    try:
            # Fetch track information
            release_id = newest_release["id"]
            release_details = musicbrainzngs.get_release_by_id(release_id, includes=["recordings"])
            tracks = release_details["release"]["medium-list"][0]["track-list"]
            album_info["tracks"] = []
            for track in tracks:
                if "title" in track:
                    album_info["tracks"].append(track["title"])
                else:
                    album_info["tracks"].append(track["recording"]["title"])

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

def download_youtube_video(url, output_folder, format, album, i = 0):
    """
    Downloads a YouTube video, converts it to WAV, splits it into segments, and saves it in the specified folder.

    Args:
        url (str): The URL of the YouTube video.
        output_folder (str): The folder where the video and its segments will be saved.
    """
    global g_download_worker
    try:
        # Create a YouTube object
        yt = YouTube(url)

        # Parse timestamps from the description
        description = yt.description
        segments = parse_timestamps(description)

        # Get video title and description
        title = yt.title

        # If the song is live, skip it
        if re.search(r'\(.*[Ll]ive.*\)', title, re.IGNORECASE):
            g_download_worker.cancel_task(title)
            return

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

        if not g_download_worker.start_task(title_str, i):
            g_download_worker.abort_task()
            return
        
        # If song is already downloaded, skip
        if os.path.exists(os.path.join(output_folder, f'{title.file_name}.{format}')):
            g_download_worker.cancel_task(title_str)
            return

        g_download_worker.increment(title_str)

        # Get the audio stream
        audio_stream = yt.streams.filter(only_audio=True).first()
        if not audio_stream:
            print(f"{COLOR_ERROR}No audio stream available for video: {yt.title}{COLOR_RESET}")
            g_download_worker.cancel_task(title_str)
            return

        # Download the audio to the output folder
        downloaded_file = audio_stream.download(output_path=output_folder)

        g_download_worker.increment(title_str)

        # Rename to use the safe title
        file_name = TEMP_NAME if segments else title.file_name
        file_name = os.path.join(output_folder, f"{file_name}.mp4")

        # If the file already exists, replace it
        if not os.path.exists(file_name):
            os.rename(downloaded_file, file_name)
        downloaded_file = file_name

        g_download_worker.increment(title_str)
        
        # if no segments are found, convert to target format and return (this is a single song video)
        if not segments:
            final_file = convert_file(downloaded_file, format)
            update_metadata(final_file, album, title)
            g_download_worker.increment(title_str)
            return final_file

        # There are segments, so proceed to split the audio into multiple songs

        # Convert the downloaded file to WAV format using ffmpeg so it can be edited
        wav_file = convert_file(downloaded_file, "wav")

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
        return None

def download_youtube_playlist(playlist_url, output_folder, format, album):
    """
    Downloads all videos in a YouTube playlist.

    Args:
        playlist_url (str): The URL of the YouTube playlist.
    """
    try:
        # Create a Playlist object
        playlist = Playlist(playlist_url)

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Iterate through all videos in the playlist
        global g_download_worker
        g_download_worker = Worker(4, len(playlist.video_urls))
        threads = []
        for i, video_url in enumerate(playlist.video_urls):
            thread = threading.Thread(target=download_youtube_video, args=(video_url, output_folder, format, album, i))
            threads.append(thread)
            thread.start()
        
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

        # Check if the URL is a playlist
        if "playlist" in url:
            # Handle playlist: use the first video's thumbnail
            playlist = Playlist(url)
            if not playlist.video_urls:
                return None
            video_url = playlist.video_urls[0]  # Use the first video in the playlist
        else:
            # Handle single video
            video_url = url

        # Create a YouTube object for the selected video
        yt = YouTube(video_url)

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

def main():
    print("Welcome to the AlbumWorks downloader!")
    print("You can download individual videos or entire playlists.")
    print()
    print("Setup:")
    
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
    ALBUM_INFO_CACHE_SIZE = 10
    
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
        album_name_input = input("Enter the album name: ").strip()
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
            album_info = get_album_info(album_name, album_artist)
            album_year = None
            album_tracks = None
            manual_entry = album_info is None or "not found" in album_info or "error" in album_info or album_info["artist"] != album_artist
            if manual_entry:
                print(f"{COLOR_INFO}Album not found.{COLOR_RESET}")
                if album_info is not None and "error" in album_info:
                    print(album_info["error"])
                print("Please enter the rest of the album info manually.")
                album_year = input("Enter the album year: ").strip()
                if len(album_year) == 0: album_year = None
                album_genre = input("Enter the album genre: ").strip()
                if len(album_genre) == 0: album_genre = None
                album_tracks = []
                while True:
                    track_input = input(f"Enter the index for track to edit, or the name of a new track to add (or leave blank to finish): ").strip()
                    if len(track_input) == 0:
                        break
                    if track_input.isdigit(): # edit existing track
                        track_index = int(track_input) - 1
                        while len(album_tracks) <= track_index:
                            album_tracks.append(None)
                        track_name = input(f"Enter the name for track {track_index + 1}: ").strip()
                        album_tracks[track_index] = track_name
                    else: # add new track
                        album_tracks.append(track_input)
            else:
                album_name = album_info["name"]
                album_artist = album_info["artist"]
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

        # # Prompt to override album information
        # overided = False
        # while True:
        #     override_choice = input("Enter the name of a value to override if desired. Press Enter to continue: ").strip().lower()
        #     if len(override_choice) == 0:
        #         break
        #     overided = True
        #     if override_choice == "artist":
        #         album.artist = input("Enter the new artist name: ").strip()
        #     elif override_choice == "title":
        #         album.name = input("Enter the new album title: ").strip()
        #     elif override_choice == "year":
        #         album.year = input("Enter the new album year: ").strip()
        #     elif override_choice == "genre":
        #         album.genre = input("Enter the new album genre: ").strip()
        #     elif override_choice == "tracks":
        #         while True:
        #             track_input = input(f"Enter the index for track to edit, or the name of a new track to add (or leave blank to finish): ").strip()
        #             if len(track_input) == 0:
        #                 break
        #             if track_input.isdigit(): # edit existing track
        #                 track_index = int(track_input) - 1
        #                 while len(album.tracks) <= track_index:
        #                     album.tracks.append(None)
        #                 track_name = input(f"Enter the name for track {track_index + 1}: ").strip()
        #                 album.tracks[track_index] = track_name
        #             else: # add new track
        #                 album.tracks.append(track_input)
        #     else:
        #         print(f"{COLOR_ERROR}Unknown value: {override_choice}{COLOR_RESET}")

        # if overided:
        #     print(f"{COLOR_INFO}Updated album information:{COLOR_RESET}")
        #     album.print()
        print()

        # Prompt for YouTube URL
        url = input("Enter the YouTube video or playlist URL: ").strip()
        if len(url) == 0:
            print(f"{COLOR_ERROR}No URL given.{COLOR_RESET}")
            continue

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
        if "playlist" in url:
            # Process as a playlist
            output_folder = download_youtube_playlist(url, album_path, format, album)
        elif "watch" in url:
            # Process as a single video
            global g_download_worker
            g_download_worker = Worker(4, 1)
            output_folder = download_youtube_video(url, album_path, format, album)
            # Get the output folder from the returned file path
            if output_folder is not None and os.path.isfile(output_folder):
                output_folder = os.path.dirname(output_folder)
        else:
            print(f"{COLOR_ERROR}Invalid URL. Please provide a valid YouTube video or playlist URL.{COLOR_RESET}")
            continue

        if output_folder is None:
            print(f"{COLOR_ERROR}Download failed.{COLOR_RESET}")
        else:
            print(f"{COLOR_SUCCESS}Download successful.{COLOR_RESET}")
            print(f"{COLOR_INFO}Files saved to: {output_folder}.{COLOR_RESET}\n")
        
        # Delete thumbnail if it was downloaded
        if album_artwork_temporary and os.path.exists(album_artwork):
            try:
                os.remove(album_artwork)
            except Exception as e:
                print(f"{COLOR_ERROR}Failed to delete thumbnail: {e}{COLOR_RESET}")
        
        # Done

if __name__ == "__main__":
    main()