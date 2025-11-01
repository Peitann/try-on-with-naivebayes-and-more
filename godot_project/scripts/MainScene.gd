extends Node2D
## Main scene for Try-On Filter application
## Displays webcam feed and applies AR filter based on face tracking data

# Communication settings
const UDP_PORT = 9999
const UDP_HOST = "127.0.0.1"

# References to child nodes
@onready var webcam_display = $WebcamDisplay
@onready var filter_overlay = $FilterOverlay
@onready var canvas_editor = $CanvasEditor
@onready var status_label = $UI/StatusLabel

# UDP socket for receiving tracking data
var udp_socket: PacketPeerUDP
var last_bbox: Dictionary = {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
var frame_size: Dictionary = {"width": 640, "height": 480}

# Webcam capture
var camera_image: Image
var camera_texture: ImageTexture

func _ready():
	print("Try-On Filter - Godot Application Starting...")
	
	# Initialize UDP socket
	setup_udp_connection()
	
	# Initialize webcam (placeholder - requires GDNative or external plugin)
	setup_webcam()
	
	# Setup UI
	update_status("Waiting for tracking data...")
	
	print("Application ready!")

func setup_udp_connection():
	"""Setup UDP socket to receive face tracking data from Python."""
	udp_socket = PacketPeerUDP.new()
	var result = udp_socket.bind(UDP_PORT, UDP_HOST)
	
	if result == OK:
		print("UDP socket bound to %s:%d" % [UDP_HOST, UDP_PORT])
	else:
		print("ERROR: Failed to bind UDP socket: ", result)

func setup_webcam():
	"""Setup webcam display (placeholder implementation)."""
	# Note: Godot doesn't have built-in webcam support
	# Options:
	# 1. Use GDNative plugin for camera access
	# 2. Receive frames from Python via shared memory or network
	# 3. Use external camera plugin from asset library
	
	# For this demo, we'll create a placeholder texture
	camera_image = Image.create(640, 480, false, Image.FORMAT_RGB8)
	camera_image.fill(Color(0.2, 0.2, 0.3))  # Dark blue placeholder
	
	camera_texture = ImageTexture.create_from_image(camera_image)
	webcam_display.texture = camera_texture
	
	print("Webcam display initialized (placeholder)")

func _process(_delta):
	"""Main update loop - receives tracking data and updates display."""
	# Receive tracking data from Python
	if udp_socket.get_available_packet_count() > 0:
		receive_tracking_data()
	
	# Update filter overlay position based on face bbox
	update_filter_position()

func receive_tracking_data():
	"""Receive and parse face tracking data from Python."""
	var packet = udp_socket.get_packet()
	var packet_string = packet.get_string_from_utf8()
	
	# Parse JSON data
	var json = JSON.new()
	var error = json.parse(packet_string)
	
	if error == OK:
		var data = json.data
		
		if data.has("bbox"):
			last_bbox = data["bbox"]
		
		if data.has("frame_size"):
			frame_size = data["frame_size"]
		
		# Update status
		var bbox_str = "BBox: (%d, %d, %d, %d)" % [
			last_bbox["x1"], last_bbox["y1"],
			last_bbox["x2"], last_bbox["y2"]
		]
		update_status("Tracking active | " + bbox_str)
	else:
		print("Error parsing JSON: ", json.get_error_message())

func update_filter_position():
	"""Update filter overlay position based on face bounding box."""
	if last_bbox["x1"] == 0 and last_bbox["y1"] == 0:
		# No face detected
		filter_overlay.visible = false
		return
	
	filter_overlay.visible = true
	
	# Calculate filter position and scale
	var bbox_center_x = (last_bbox["x1"] + last_bbox["x2"]) / 2.0
	var bbox_center_y = (last_bbox["y1"] + last_bbox["y2"]) / 2.0
	var bbox_width = last_bbox["x2"] - last_bbox["x1"]
	var bbox_height = last_bbox["y2"] - last_bbox["y1"]
	
	# Map from camera coordinates to screen coordinates
	var screen_x = (bbox_center_x / frame_size["width"]) * webcam_display.size.x
	var screen_y = (bbox_center_y / frame_size["height"]) * webcam_display.size.y
	
	# Update filter overlay position
	filter_overlay.position = Vector2(screen_x, screen_y)
	
	# Scale filter to match face size
	var scale_x = bbox_width / 200.0  # Adjust divisor based on filter size
	var scale_y = bbox_height / 200.0
	filter_overlay.scale = Vector2(scale_x, scale_y)

func update_status(message: String):
	"""Update status label text."""
	if status_label:
		status_label.text = message

func _exit_tree():
	"""Cleanup on exit."""
	if udp_socket:
		udp_socket.close()
	print("Application shutting down...")
