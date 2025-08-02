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
import sys

FFMPEG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "ffmpeg", "bin", "ffmpeg.exe")
OUTPUT_PATH = os.path.join(os.path.expanduser("~"), "Music")
FORMATS = ["mp3", "wav", "mp4a"]
TEMP_NAME = "temp"
THUMBNAIL_NAME = "thumbnail"

COLOR_ERROR = "\033[91m"
COLOR_SUCCESS = "\033[92m"
COLOR_INFO = "\033[90m"
COLOR_RESET = "\033[0m"

def find_closest_string_index(target, string_list):
    """
    Finds the index of the closest string in a list to the given target string.

    Args:
        target (str): The string to compare against.
        string_list (list): A list of strings to search.

    Returns:
        int: The index of the closest matching string, or -1 if no match is found.
    """
    matches = difflib.get_close_matches(target, string_list, n=1, cutoff=0.8)
    if matches:
        closest_match = matches[0]
        return string_list.index(closest_match)
    return -1

def set_title(artist = None, album = None, song = None):
    """Sets the console title to the current album, artist, and song."""
    if artist is None:
        artist = ""
    else:
        artist = f' - {artist}'
    if album is None:
        album = ""
    else:
        album = f' - {album}'
    if song is None:
        song = ""
    else:
        song = f' - {song}'
    os.system(f'title "AlbumWorks{artist}{album}{song}"')

class Album:
    def __init__(self, name, artist, year, genre, tracks, artwork):
        self.name = name
        self.artist = artist
        self.year = year
        self.genre = genre
        self.artwork = artwork
        self.tracks = tracks if tracks else []
        self.lowered_tracks = [track.lower() for track in tracks] if tracks else []

    def get_track_index(self, name):
        """
        Finds the index of a track by its name.

        Args:
            name (str): The name of the track.

        Returns:
            int: The index of the track, or -1 if not found.
        """
        index = find_closest_string_index(name.lower(), self.lowered_tracks)
        if index != -1:
            # Existing track found
            return index
        # Add new track
        index = len(self.tracks)
        self.tracks.append(name)
        self.lowered_tracks.append(name.lower())
        return index
    
    def get_folder_name(self):
        """
        Returns a sanitized folder name for the album.

        Returns:
            str: The sanitized folder name.
        """
        return fix_path(f"{self.artist} - {self.name}")
    
def update_metadata(file_path, title, artist, album, track_number, year=None, genre=None, artwork=None):
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
        print(f"{COLOR_ERROR}Failed to update metadata for {file_path}. Error: {e}{COLOR_RESET}")

def update_metadata_with_album(file_path, album):
    """
    Updates the metadata of an audio file.

    Args:
        file_path (str): Path to the audio file.
        album (Album): Album object containing metadata.
    """
    # Get title from the file name
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Get the track index
    track_index = album.get_track_index(file_name)

    # Now get the actual track title
    title = album.tracks[track_index]

    update_metadata(file_path, title, album.artist, album.name, track_index + 1, album.year, album.genre, album.artwork)

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
    NUMBER_PATTERN = r"(?:\d+\.?\)?)?\s*"
    NAME_PATTERN = r"(.+)"
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
        total_seconds = sum(x * 60 ** i for i, x in enumerate(reversed(time_parts)))
        segments.append((total_seconds, name.strip()))
    return segments

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

def split_audio(file_path, segments, output_folder, format, album):
    """
    Splits the audio file into segments based on timestamps.

    Args:
        file_path (str): Path to the audio file.
        segments (list): List of tuples (start_time, segment_name).
        output_folder (str): Folder to save the split audio files.
    """
    used = set()
    with wave.open(file_path, "rb") as audio:
        params = audio.getparams()
        frame_rate = params.framerate

        for i, (start_time, name) in enumerate(segments):
            # Determine the start and end frames
            start_frame = int(start_time * frame_rate)
            end_frame = int(segments[i + 1][0] * frame_rate) if i + 1 < len(segments) else params.nframes

            # Set the position and read frames
            audio.setpos(start_frame)
            frames = audio.readframes(end_frame - start_frame)

            # Save the segment with the given name
            title = fix_title(name, album)
            track_index = album.get_track_index(title)

            # If a duplicate, increment the title
            while track_index in used:
                title = increment_or_add_suffix(title)
                track_index = album.get_track_index(title)

            used.add(track_index)

            print(f"\t{track_index + 1}) {title}")
            set_title(album.artist, album.name, title)
            output_path = os.path.join(output_folder, f"{title}.wav")
            with wave.open(output_path, "wb") as segment:
                segment.setparams(params)
                segment.writeframes(frames)
            final_file = convert_file(output_path, format)
            update_metadata_with_album(final_file, album)

def convert_file(path, format):
    """
    Converts a file to the specified format.

    Args:
        path (str): Path to the file to convert.
        format (str): Desired output format (e.g., 'mp3', 'wav', 'mp4a').

    Returns:
        str: Path to the converted file.
    """
    # Get the base name and current extension of the file
    base, ext = os.path.splitext(path)
    current_format = ext[1:]  # Remove the leading dot from the extension
    target_path = f"{base}.{format}"  # Target file path with the new format

    # If the file is already in the desired format, return the original path
    if current_format == format:
        return path

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
    # Set up MusicBrainz client
    musicbrainzngs.set_useragent("AlbumWorks", "1.0", "https://example.com")

    try:
        # Search for the album by name and artist
        result = musicbrainzngs.search_releases(release=album_name, artist=artist_name, limit=1)
        if result["release-list"]:
            release = result["release-list"][0]
            album_info = {
                "name": release["title"],
                "artist": release["artist-credit"][0]["artist"]["name"],
                "release_date": release.get("date", "Unknown"),
                "tracks": [],
                "url": f"https://musicbrainz.org/release/{release['id']}"
            }

            # Fetch track information
            release_id = release["id"]
            release_details = musicbrainzngs.get_release_by_id(release_id, includes=["recordings"])
            tracks = release_details["release"]["medium-list"][0]["track-list"]
            album_info["tracks"] = [track["recording"]["title"] for track in tracks]

            return album_info
        else:
            return {"not found": "Album not found"}
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

def fix_path(path):
    # Remove illegal characters for file and folder names
    path = re.sub(r'[<>:"/\\|?*]', '', path)
    return path.strip()

def fix_title(title, album = None):
    """
    Cleans up the YouTube title by removing unwanted text and illegal characters.

    Args:
        title (str): The original YouTube title.

    Returns:
        str: The sanitized title.
    """
    # Remove "(...)" text"
    title = re.sub(r'(\(.*\))', '', title, flags=re.IGNORECASE)

    # Remove any common words or phrases "OST"
    title = re.sub(r'\b(OST|Lyrics)\b', '', title, flags=re.IGNORECASE)

    # Remove anything past '|'
    title = re.sub(r'\|.*$', '', title, flags=re.IGNORECASE)

    # Remove any mention of the artist name
    if album:
        title = re.sub(rf'\b{re.escape(album.artist)}\b', '', title, flags=re.IGNORECASE)

    # Remove " - "or " | ", etc.
    title = re.sub(r' ?[-|] ?', '', title)

    # Remove leading or trailing whitespace and punctuation
    title = re.sub(r'^[^a-zA-Z0-9\(]+|[^a-zA-Z0-9\)]+$', '', title)

    # If no album, done here
    if not album:
        return fix_path(title)

    # Remove any mention of the album name
    temp = re.sub(rf'\b{re.escape(album.name)}\b', '', title, flags=re.IGNORECASE).strip()

    # If the title is empty, use the album name
    empty_pattern = r'^[^a-zA-Z0-9]*$'
    if re.fullmatch(empty_pattern, temp):
        title = album.name
    else:
        title = temp

    # If there is a match to a track, use that instead
    track_index = album.get_track_index(title)
    if track_index != -1:
        title = album.tracks[track_index]

    return fix_path(title)

def download_youtube_video(url, output_folder, format, album):
    """
    Downloads a YouTube video, converts it to WAV, splits it into segments, and saves it in the specified folder.

    Args:
        url (str): The URL of the YouTube video.
        output_folder (str): The folder where the video and its segments will be saved.
    """
    try:
        # Create a YouTube object
        yt = YouTube(url)

        # Parse timestamps from the description
        description = yt.description
        segments = parse_timestamps(description)

        # Get video title and description
        title = yt.title

        # Print index and title, if no segments found
        if not segments:
            title = fix_title(yt.title, album)
            track_index = album.get_track_index(title)
            print(f"\t{track_index + 1}) {title}")
            set_title(album.artist, album.name, title)
        else:
            title = fix_title(yt.title, album)

        # Get the audio stream
        audio_stream = yt.streams.filter(only_audio=True).first()
        if not audio_stream:
            print(f"{COLOR_ERROR}No audio stream available for video: {yt.title}{COLOR_RESET}")
            return

        # Download the audio to the output folder
        downloaded_file = audio_stream.download(output_path=output_folder)

        # Rename to use the safe title
        file_name = TEMP_NAME if segments else title
        file_name = os.path.join(output_folder, f"{file_name}.mp4")
        os.rename(downloaded_file, file_name)
        downloaded_file = file_name

        # if no segments are found, convert to target format and return
        if not segments:
            final_file = convert_file(downloaded_file, format)
            update_metadata_with_album(final_file, album)
            return final_file

        # Convert the downloaded file to WAV format using ffmpeg so it can be edited
        wav_file = convert_file(downloaded_file, "wav")

        # Create a folder for the video's segments
        os.makedirs(output_folder, exist_ok=True)

        # Split the audio into segments
        split_audio(wav_file, segments, output_folder, format, album)

        # Delete the temporary WAV file
        os.remove(wav_file)

        return output_folder
    except Exception as e:
        print(f"{COLOR_ERROR}An error occurred while processing video: {url}. Error: {e}{COLOR_RESET}")
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
        for video_url in playlist.video_urls:
            download_youtube_video(video_url, output_folder, format, album)

        print(f"{COLOR_SUCCESS}Playlist downloaded successfully: {playlist.title}{COLOR_RESET}")

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

        # Download the image
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(image_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            return image_path
        else:
            print(f"{COLOR_ERROR}Failed to download image. HTTP Status Code: {response.status_code}{COLOR_RESET}")
            return None
        
        return image_path

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

def main():
    os.system("cls")
    print("Welcome to the AlbumWorks downloader!")
    print("You can download individual videos or entire playlists.")
    print("")
    while True:
        set_title()
        # Prompt for album information
        album_artist = input("Enter the album artist: ").strip()
        if len(album_artist) == 0:
            print(f"{COLOR_ERROR}No album artist given.{COLOR_RESET}")
            return
        album_name = input("Enter the album name: ").strip()
        if len(album_name) == 0:
            print(f"{COLOR_ERROR}No album name given.{COLOR_RESET}")
            return
        print(f"{COLOR_INFO}Retrieving album information...{COLOR_RESET}")
        album_info = get_album_info(album_name, album_artist)
        album_year = None
        album_tracks = None
        if "not found" in album_info or "error" in album_info or album_info["artist"] != album_artist:
            print(f"{COLOR_INFO}Album not found.{COLOR_RESET}")
            if "error" in album_info:
                print(album_info["error"])
            print("Please enter the rest of the album info manually.")
            album_year = input("Enter the album year: ").strip()
            if len(album_year) == 0: album_year = None
            album_genre = input("Enter the album genre: ").strip()
            if len(album_genre) == 0: album_genre = None
        else:
            album_year = album_info["release_date"][:4]
            album_genre = album_info["genre"]
            album_tracks = album_info["tracks"]
            print(f"{COLOR_SUCCESS}Done.{COLOR_RESET}")

        # Prompt for output format
        formats_list = ", ".join(FORMATS)
        format = input(f"Enter the desired output format ({formats_list}): ").strip().lower()
        if len(format) == 0:
            print(f"{COLOR_INFO}Defaulting to {FORMATS[0]}.{COLOR_RESET}")
            format = FORMATS[0]
        if format not in FORMATS:
            print(f"{COLOR_ERROR}Invalid format. Supported formats are: {formats_list}{COLOR_RESET}")
            return

        # Prompt for YouTube URL
        url = input("Enter the YouTube video or playlist URL: ").strip()
        if len(url) == 0:
            print(f"{COLOR_ERROR}No URL given.{COLOR_RESET}")
            return
        
        # Prompt for artwork, if user wants to override
        album_artwork = input("Enter the path to the album artwork (or leave blank to use the thumbnail): ").strip()
        album_artwork_temporary = len(album_artwork) == 0
        if not album_artwork_temporary:
            # Use existing file or url
            if not os.path.exists(album_artwork):
                # if no file exists, assume it's a URL and download it
                album_artwork = download_image(album_artwork, OUTPUT_PATH, THUMBNAIL_NAME)

            # Check if the file exists, if not, print an error
            if not os.path.exists(album_artwork):
                print(f"{COLOR_ERROR}Artwork path does not exist.{COLOR_RESET}")
                return
        else:
            # Download the thumbnail
            print(f"{COLOR_INFO}Downloading thumbnail...{COLOR_RESET}")

            # Create the output path for the thumbnail
            os.makedirs(OUTPUT_PATH, exist_ok=True)

            # Download and use the thumbnail
            album_artwork = download_youtube_thumbnail(url, OUTPUT_PATH, THUMBNAIL_NAME)

        album = Album(album_name, album_artist, album_year, album_genre, album_tracks, album_artwork)

        # Prompt for output folder
        output_folder = input(f"Enter the output folder (default: {OUTPUT_PATH}): ").strip()
        if len(output_folder) == 0:
            output_folder = OUTPUT_PATH
        if not os.path.exists(output_folder):
            print(f"{COLOR_ERROR}Output folder does not exist.{COLOR_RESET}")
            return

        print("")
        print(f"\t\t{album.name}:")
        set_title(album.artist, album.name)
        album_path = os.path.join(output_folder, album.get_folder_name())
        if "playlist" in url:
            # Process as a playlist
            output_folder = download_youtube_playlist(url, album_path, format, album)
        elif "watch" in url:
            # Process as a single video
            output_folder = download_youtube_video(url, album_path, format, album)
        else:
            print(f"{COLOR_ERROR}Invalid URL. Please provide a valid YouTube video or playlist URL.{COLOR_RESET}")
            return
        set_title(album.artist, album.name)
        
        # Delete thumbnail if it was downloaded
        if album_artwork_temporary and album_artwork:
            try:
                os.remove(album_artwork)
            except Exception as e:
                print(f"{COLOR_ERROR}Failed to delete thumbnail: {e}{COLOR_RESET}")
        
        # Done
        print("")
        print(f"Files saved to: {output_folder}\n")
        print('Press Enter to run AlbumWorks again or type "exit" to quit.')
        choice = input()
        if choice.lower() == "exit":
            break
        print("-" * os.get_terminal_size().columns)

if __name__ == "__main__":
    main()