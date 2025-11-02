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
@onready var bbox_rect = $BBoxRect  # Visual bounding box indicator

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
	
	# Setup bounding box rect if not in scene
	if not has_node("BBoxRect"):
		var rect = ColorRect.new()
		rect.name = "BBoxRect"
		rect.color = Color(0, 1, 0, 0.5)  # Green with 50% opacity (more visible)
		rect.visible = false
		rect.z_index = 10  # Put in front of everything
		add_child(rect)
		bbox_rect = rect
		print("BBoxRect created programmatically")
	else:
		# Make sure existing rect is visible and in front
		bbox_rect.z_index = 10
		bbox_rect.color = Color(0, 1, 0, 0.5)
		print("BBoxRect found in scene")
	
	# Setup UI
	update_status("Waiting for tracking data...")
	
	# TEST: Show a static green box for debugging
	if bbox_rect:
		await get_tree().create_timer(1.0).timeout  # Wait 1 second
		bbox_rect.position = Vector2(400, 200)  # Static position
		bbox_rect.size = Vector2(200, 150)  # Static size
		bbox_rect.visible = true
		print("TEST: Static green box should be visible at (400, 200)")
	
	print("Application ready!")
	print("BBoxRect exists: ", bbox_rect != null)

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
	# For now, create a visible placeholder to show camera area
	
	# Create a larger, visible placeholder
	camera_image = Image.create(640, 480, false, Image.FORMAT_RGB8)
	
	# Create a gradient pattern so it's visible
	for y in range(480):
		for x in range(640):
			var r = float(x) / 640.0
			var g = float(y) / 480.0
			var b = 0.5
			camera_image.set_pixel(x, y, Color(r, g, b))
	
	camera_texture = ImageTexture.create_from_image(camera_image)
	webcam_display.texture = camera_texture
	
	# Make sure it's visible and centered
	webcam_display.position = Vector2(640, 360)
	webcam_display.centered = true
	
	print("Webcam display initialized (placeholder gradient)")
	print("Webcam display position: ", webcam_display.position)
	print("Webcam texture size: ", camera_texture.get_width(), "x", camera_texture.get_height())
	
	# Draw border around webcam area for reference
	queue_redraw()

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
	# Check if face is detected (non-zero bbox)
	var has_face = not (last_bbox["x1"] == 0 and last_bbox["y1"] == 0 and last_bbox["x2"] == 0 and last_bbox["y2"] == 0)
	
	if not has_face:
		# No face detected
		filter_overlay.visible = false
		if bbox_rect:
			bbox_rect.visible = false
		return
	
	filter_overlay.visible = true
	
	# Calculate filter position and scale
	var bbox_center_x = (last_bbox["x1"] + last_bbox["x2"]) / 2.0
	var bbox_center_y = (last_bbox["y1"] + last_bbox["y2"]) / 2.0
	var bbox_width = last_bbox["x2"] - last_bbox["x1"]
	var bbox_height = last_bbox["y2"] - last_bbox["y1"]
	
	# Get webcam display size (Sprite2D uses texture size)
	var display_width = frame_size["width"]
	var display_height = frame_size["height"]
	
	if webcam_display.texture:
		display_width = webcam_display.texture.get_width()
		display_height = webcam_display.texture.get_height()
	
	# Map from camera coordinates to screen coordinates
	# Webcam is centered at (640, 360), so adjust coordinates
	var screen_x = webcam_display.position.x - display_width/2.0 + bbox_center_x
	var screen_y = webcam_display.position.y - display_height/2.0 + bbox_center_y
	
	# Update filter overlay position
	filter_overlay.position = Vector2(screen_x, screen_y)
	
	# Scale filter to match face size
	var scale_x = bbox_width / 200.0  # Adjust divisor based on filter size
	var scale_y = bbox_height / 200.0
	filter_overlay.scale = Vector2(scale_x, scale_y)
	
	# Update visual bounding box
	if bbox_rect:
		bbox_rect.visible = true
		var bbox_screen_x1 = webcam_display.position.x - display_width/2.0 + last_bbox["x1"]
		var bbox_screen_y1 = webcam_display.position.y - display_height/2.0 + last_bbox["y1"]
		bbox_rect.position = Vector2(bbox_screen_x1, bbox_screen_y1)
		bbox_rect.size = Vector2(bbox_width, bbox_height)
		
		# Debug print (every 60 frames = ~1 second)
		if Engine.get_process_frames() % 60 == 0:
			print("BBox Rect - Pos: ", bbox_rect.position, " Size: ", bbox_rect.size, " Visible: ", bbox_rect.visible)

func update_status(message: String):
	"""Update status label text."""
	if status_label:
		status_label.text = message

func _exit_tree():
	"""Cleanup on exit."""
	if udp_socket:
		udp_socket.close()
	print("Application shutting down...")

func _draw():
	"""Draw debug visualizations."""
	# Draw border around webcam area
	var cam_pos = webcam_display.position
	var cam_w = 640
	var cam_h = 480
	var top_left = Vector2(cam_pos.x - cam_w/2, cam_pos.y - cam_h/2)
	var rect = Rect2(top_left, Vector2(cam_w, cam_h))
	draw_rect(rect, Color.WHITE, false, 2.0)  # White border, 2px thick
