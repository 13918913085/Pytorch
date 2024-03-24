### reshape()和view()的区别
- reshape()函数返回一个新的张量，而view()函数返回一个与原始张量共享存储空间的张量。这意味着，当你使用reshape()函数改变张量形状时，会创建一个新的张量对象，而原始张量对象不会改变。而当你使用view()函数改变张量形状时，会返回一个新的张量对象，但是它与原始张量对象共享存储空间，因此对新张量的修改也会影响原始张量。
- reshape()函数可以处理任意形状的张量，而view()函数只能处理连续的张量。如果你尝试使用view()函数处理非连续的张量，会引发RuntimeError异常 解决方法 y.contiguous().view(9)

[Aladdin Persson](https://www.youtube.com/watch?v=x9JiIFvlUwk)
