# imports necessary to draw and randomize
from PIL import Image, ImageDraw
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
    self.shapes = ['circle']

    # draw a random shape
    self.draw_random_shape()
  
  # draw a random shape -- for now just cirlce
  def draw_random_shape(self):
    draw_me = random.choice(self.shapes)
    if draw_me == 'circle':
      self.draw_circle()

  # draw a circle
  def draw_circle(self):
    # get the max diameter so the circle doesn't go out of frame
    max_diameter = min(self.image_size) // 2

    # make the diameter be a random number between 50 and the maximum allowed
    diameter = random.randint(20, max_diameter)

    # set the upper bounds to be random, while staying in the bounds of the frame
    upper_left_x = random.randint(0, self.image_size[0] - diameter)
    upper_left_y = random.randint(0, self.image_size[1] - diameter)

    # draw the circle, both outlined and filled in, and save the images
    self.draw.ellipse((upper_left_x, upper_left_y, upper_left_x + diameter, upper_left_y + diameter), outline="black", width=2)
    self.im.save('./outline.jpg')
    self.draw.ellipse((upper_left_x, upper_left_y, upper_left_x + diameter, upper_left_y + diameter), outline="black", fill=(0, 0, 0), width=2)
    self.im.save('./filled.jpg')