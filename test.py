import tensorflow as tf


def create_graph():
    with open('./model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        return tf.import_graph_def(graph_def,return_elements=["pre:0","input_x:0","keep:0"])


def name2vec(name):
    ans = ""
    for i in name:
        ans += str(i)
    return ans


output,x_,keep= create_graph()
sess = tf.Session()


def pre(image_data):
    predictions = sess.run(output, {x_: image_data,keep:1})
    vec = predictions[0].tolist()
    return name2vec(vec)
