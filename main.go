package main

import (
	"image"
	_ "image/jpeg"
	"os"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var graph *tf.Graph
var session *tf.Session
var list chan *tf.Tensor = make(chan *tf.Tensor, 10)
var keep *tf.Tensor

func init() {
	saveModel, err := tf.LoadSavedModel("./export", []string{"input_x", "pre", "keep"}, nil)
	if err != nil {
		panic(err)
	}

	graph = saveModel.Graph
	session = saveModel.Session
	keep, _ = tf.NewTensor(float32(1.0))
}

func main() {
	// go Start()
	image_, err := LoadImage("./val/0048.jpg")
	if err != nil {
		panic(err)
	}
	tensor, err := tf.NewTensor(image_)
	if err != nil {
		panic(err)
	}
	pre(tensor)
}

func Start() {
	for {
		tensor := <-list
		pre(tensor)
	}
}

func pre(tensor *tf.Tensor) {
	session.Run(map[tf.Output]*tf.Tensor{
		graph.Operation("input_x").Output(0): tensor,
		graph.Operation("keep").Output(0):    keep,
	},
		[]tf.Output{
			graph.Operation("pre").Output(0),
		}, nil)
}

func LoadImage(path string) ([][]uint8, error) {
	reader, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	picture, _, err := image.Decode(reader)
	if err != nil {
		return nil, err
	}

	image_ := make([][]uint8, picture.Bounds().Dy())
	for i := 0; i < picture.Bounds().Dy(); i++ {
		image_[i] = make([]uint8, picture.Bounds().Dx())
	}

	for y := picture.Bounds().Min.Y; y < picture.Bounds().Max.Y; y++ {
		for x := picture.Bounds().Min.X; x < picture.Bounds().Max.X; x++ {
			r, g, b, _ := picture.At(x, y).RGBA()
			gray := (19595*r + 38470*g + 7471*b + 1<<15) >> 24
			if gray > 170 {
				image_[y][x] = 255
			} else {
				image_[y][x] = 0
			}
		}
	}
	return image_, nil
}
