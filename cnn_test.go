package main

import (
	"testing"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var tensor *tf.Tensor

func init() {
	image_, err := LoadImage("./val/0048.jpg")
	if err != nil {
		panic(err)
	}
	tensor, _ = tf.NewTensor(image_)
}

func BenchmarkPre(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pre(tensor)
	}
}
