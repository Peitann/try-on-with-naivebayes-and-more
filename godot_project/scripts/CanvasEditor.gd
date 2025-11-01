extends Control
## Canvas editor for creating stickers and doodles

signal filter_updated(filter_data)

# Drawing state
var is_drawing: bool = false
var current_stroke: PackedVector2Array = []
var strokes: Array = []
var current_color: Color = Color.RED
var stroke_width: float = 3.0

# References
@onready var canvas = $DrawingCanvas
@onready var color_picker = $UI/ColorPicker
@onready var clear_button = $UI/ClearButton
@onready var save_button = $UI/SaveButton

func _ready():
	print("Canvas editor initialized")
	
	# Connect signals
	if clear_button:
		clear_button.pressed.connect(_on_clear_pressed)
	if save_button:
		save_button.pressed.connect(_on_save_pressed)
	if color_picker:
		color_picker.color_changed.connect(_on_color_changed)

func _input(event):
	"""Handle drawing input."""
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT:
			if event.pressed:
				# Start drawing
				is_drawing = true
				current_stroke = PackedVector2Array()
				current_stroke.append(event.position)
			else:
				# End drawing
				is_drawing = false
				if current_stroke.size() > 1:
					strokes.append({
						"points": current_stroke,
						"color": current_color,
						"width": stroke_width
					})
					emit_signal("filter_updated", strokes)
				current_stroke = PackedVector2Array()
	
	elif event is InputEventMouseMotion:
		if is_drawing:
			current_stroke.append(event.position)
			queue_redraw()  # Request redraw

func _draw():
	"""Draw all strokes and current stroke."""
	# Draw saved strokes
	for stroke_data in strokes:
		var points = stroke_data["points"]
		var color = stroke_data["color"]
		var width = stroke_data["width"]
		
		for i in range(points.size() - 1):
			draw_line(points[i], points[i + 1], color, width)
	
	# Draw current stroke
	if is_drawing and current_stroke.size() > 1:
		for i in range(current_stroke.size() - 1):
			draw_line(current_stroke[i], current_stroke[i + 1], current_color, stroke_width)

func _on_clear_pressed():
	"""Clear all strokes."""
	strokes.clear()
	current_stroke = PackedVector2Array()
	queue_redraw()
	emit_signal("filter_updated", strokes)
	print("Canvas cleared")

func _on_save_pressed():
	"""Save current filter design."""
	print("Filter saved with %d strokes" % strokes.size())
	# TODO: Implement filter save/load functionality

func _on_color_changed(new_color: Color):
	"""Update drawing color."""
	current_color = new_color
	print("Drawing color changed to: ", new_color)

func export_filter_data() -> Dictionary:
	"""Export filter data for application."""
	return {
		"strokes": strokes,
		"timestamp": Time.get_ticks_msec()
	}
