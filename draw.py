# imports necessary to draw and randomize
from PIL import Image, ImageDraw
import math
import random

# class to draw an outline and filled-in shape
class DrawAShape:
  # initialize the shape with an image size
  def __init__(self, image_size=(224, 224)):
    self.image_size = image_size

    # create image outline with a white background
    self.im = Image.new('RGB', self.image_size, (255, 255, 255))

    # initialize ImageDraw
    self.draw = ImageDraw.Draw(self.im)

    # create a list of shapes to choose from -- just circle for now
    self.shapes = ['circle', 'polygon', 'polygon','polygon', 'dot', 'line', 'rectangle', 'star']

    # draw a random shape
    self.draw_random_shape()

  # draw a random shape
  def draw_random_shape(self):
    draw_me = random.choice(self.shapes)
    if draw_me == 'circle':
      self.draw_circle()
    elif draw_me == 'polygon':
      self.draw_polygon()
    elif draw_me == 'dot':
      self.draw_dot()
    elif draw_me == 'line':
      self.draw_line()
    elif draw_me == 'rectangle':
      self.draw_rectangle()
    elif draw_me == 'star':
      self.draw_star()

  # draw a circle
  def draw_circle(self):
    # get the max diameter so the circle doesn't go out of frame
    max_diameter = min(self.image_size) // 2

    # make the diameter be a random number between 50 and the maximum allowed
    diameter = random.randint(50, max_diameter)

    # set the upper bounds to be random, while staying in the bounds of the frame
    upper_left_x = random.randint(10, self.image_size[0] - diameter - 10)
    upper_left_y = random.randint(10, self.image_size[1] - diameter - 10)

    # draw the circle, both outlined and filled in, and save the images
    self.draw.ellipse((upper_left_x, upper_left_y, upper_left_x + diameter, upper_left_y + diameter), outline="black", width=2)
    self.im.save('./outline.jpg')
    self.draw.ellipse((upper_left_x, upper_left_y, upper_left_x + diameter, upper_left_y + diameter), outline="black", fill=(0, 0, 0), width=2)
    self.im.save('./filled.jpg')
  
  # draw a polygon
  def draw_polygon(self):
    # get the number of sides for the polygon
    side = random.randint(3, 8)

    # set the center of the polygon to be a random point *in* the frame
    safe_margin = min(self.image_size) // 4
    center = (random.randint(safe_margin, self.image_size[0] - safe_margin),
              random.randint(safe_margin, self.image_size[1] - safe_margin))
    radius = safe_margin // 2

    # get the xy coordinates for the polygon
    xy = [
        ((math.cos(th) + 1) * radius + center[0],
          (math.sin(th) + 1) * radius + center[1])
        for th in [i * (2 * math.pi) / side for i in range(side)]
    ]

    # draw the polygon, both outlined and filled in, and save the images
    self.draw.polygon(xy, outline="black")
    self.im.save('./outline.jpg')
    self.draw.polygon(xy, outline="black", fill="black")
    self.im.save('./filled.jpg')

  # draw a dot
  def draw_dot(self):
    # set the center of the dot to be a random point the frame
    center = (random.randint(10, self.image_size[0] - 10),
              random.randint(10, self.image_size[1] - 10))
    
    # draw the dot, both outlined and filled in, which are the same thing, and save the images
    self.draw.ellipse((center[0] - 5, center[1] - 5, center[0] + 5, center[1] + 5), outline="black", fill="black")
    self.im.save('./outline.jpg')
    self.draw.ellipse((center[0] - 5, center[1] - 5, center[0] + 5, center[1] + 5), outline="black", fill="black")
    self.im.save('./filled.jpg')
  
  # draw a line
  def draw_line(self):
    # set the start and end points of the line to be random points in the frame
    start = (random.randint(10, self.image_size[0] - 10), random.randint(10, self.image_size[1] - 10))
    end = (random.randint(10, self.image_size[0] - 10), random.randint(10, self.image_size[1] - 10))

    # draw the line and save the image
    self.draw.line([start, end], fill="black", width=2)
    self.im.save('./outline.jpg')
    self.im.save('./filled.jpg')

  # draw a rectangle
  def draw_rectangle(self):
    # randomly choose the first corner of the rectangle
    corner1_x = random.randint(0, self.image_size[0] - 10) 
    corner1_y = random.randint(0, self.image_size[1] - 10)

    # determine maximum possible width and height from the first corner
    max_width = self.image_size[0] - corner1_x
    max_height = self.image_size[1] - corner1_y

    # choose random width and height, ensuring the rectangle will fit within the image
    width = random.randint(10, max_width)
    height = random.randint(10, max_height)

    # calculate the second corner of the rectangle based on the chosen width and height
    corner2_x = corner1_x + width
    corner2_y = corner1_y + height

    # draw the rectangle outline and then fill it in, saving each version
    self.draw.rectangle([corner1_x, corner1_y, corner2_x, corner2_y], outline="black", width=2)
    self.im.save('./outline.jpg')
    self.draw.rectangle([corner1_x, corner1_y, corner2_x, corner2_y], outline="black", fill="black", width=2)
    self.im.save('./filled.jpg')

  # draw a star
  def draw_star(self):
    # calculate a safe margin for the star to fit within the image
    safe_margin = min(self.image_size) // 4
    max_outer_radius = safe_margin // 2

    # define a random center within bounds that account for the outer radius
    center_x = random.randint(max_outer_radius, self.image_size[0] - max_outer_radius)
    center_y = random.randint(max_outer_radius, self.image_size[1] - max_outer_radius)

    # define outer and inner radii of the star
    outer_radius = random.randint(max_outer_radius // 2, max_outer_radius)
    inner_radius = outer_radius // 2

    # calculate vertices for outer and inner points
    points = []
    for i in range(10):
        angle = math.pi / 2 + (math.pi * i / 5)
        if i % 2 == 0:
            # Outer point
            x = center_x + int(outer_radius * math.cos(angle))
            y = center_y + int(outer_radius * math.sin(angle))
        else:
            # Inner point
            x = center_x + int(inner_radius * math.cos(angle))
            y = center_y + int(inner_radius * math.sin(angle))
        points.append((x, y))
    
    # draw the star, both outlined and filled in, and save the images
    self.draw.polygon(points, outline="black")
    self.im.save('./outline.jpg')
    self.draw.polygon(points, outline="black", fill="black")
    self.im.save('./filled.jpg')