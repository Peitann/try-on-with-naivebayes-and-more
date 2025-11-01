extends Sprite2D
## Filter overlay that displays stickers and doodles on the face

# Filter elements (stickers, doodles)
var filter_elements: Array = []

func _ready():
	print("Filter overlay initialized")
	
	# Example: Load a test sticker
	#load_sticker("res://assets/stickers/sunglasses.png")

func load_sticker(sticker_path: String):
	"""Load and add a sticker to the filter."""
	var sticker_texture = load(sticker_path)
	if sticker_texture:
		var sticker_sprite = Sprite2D.new()
		sticker_sprite.texture = sticker_texture
		add_child(sticker_sprite)
		filter_elements.append(sticker_sprite)
		print("Sticker loaded: ", sticker_path)
	else:
		print("Failed to load sticker: ", sticker_path)

func add_doodle(points: PackedVector2Array, color: Color, width: float):
	"""Add a doodle line to the filter."""
	var line = Line2D.new()
	line.points = points
	line.default_color = color
	line.width = width
	add_child(line)
	filter_elements.append(line)
	print("Doodle added with %d points" % points.size())

func clear_filter():
	"""Clear all filter elements."""
	for element in filter_elements:
		element.queue_free()
	filter_elements.clear()
	print("Filter cleared")

func _process(_delta):
	pass
