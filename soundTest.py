import vlc


def play_audio(video_url):
    # VLC options to disable video output
    options = [
        '--no-video',  # Disable video output
        '--aout=alsa',  # Set audio output module (default might work)
    ]

    # Create a VLC media instance with the options
    instance = vlc.Instance(options)  # Initialize the VLC instance with options
    player = instance.media_player_new()  # Create the MediaPlayer from the instance

    # Set media to the player
    media = vlc.Media(video_url)
    player.set_media(media)

    # Start playing the audio
    player.play()

    # Keep playing until user decides to stop
    input("Press Enter to stop playing audio...")

    # Stop the player
    player.stop()
play_audio('http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4')