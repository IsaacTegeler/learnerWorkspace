# learnerWorkspace
<html style="font-family:arial; color:#404040; line-height: 1.8;">

<div  style="width:800px;">
<H1><center>Keras and Machine Learing</center></H1>
<hr>
<div style="width:500px;">
<p>Tutorias used: 
<br>
<center><a href="https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html" target="_blank"> Visualizing Layers</a><br>
     <a href="https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html" target="_blank">Image augmentation</a></center> 
</p>
<p>Interesting Localization paper: <br>
<center><a href="http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf" target="_blank">Deep Features for Discriminative Localization</a></center>
</p>
<p>E-mail: isaac.tegeler@gmail.com    <a href="https://github.com/IsaacIsOkay/learnerWorkspace">Github</a></p>
</div>
<hr>   
<body>
<div>
<H3 style="color:#000000;"> Key Terms </H3>
<ul>
    <li><b>Epoch</b> - one pass through the training data set followed by testing on validation data, should be set as the size of you dataset for training</li>
    <br>
    <li><b>Batch</b> - the number of samples from training set to train network 
                        (forward pass and backpropigation) the larger this number the more memory taken by the system. 
                        The GPU only has so much space, and I found that making this too large causes memory shortage problems.         
                        Increasing batch size increases the speed at which the network is trained.<span style="color:red"> Start with</span> 
                        a batch size of arround 100. And make the batches a multiple of the number of traing images you have.</li>
    <br>
    <li><b>Input shape</b> this is the dimensions of your input image. Images are pulled in by keras's image handeler as rgb images so they have a depth of 3.
</ul>
</div>

<div>
    <H4 style="color:#000000;">Make sure to save your weights after training</H4>
    <center style="font-family:courier;">model.save_weights("fileName.hp5")</center>
</div>

<hr>
<div>
    <H3 id="input shape" style="color:#000000;"> Input Dimensions </H3>
    <p style="text-indent:2.0em;">In keras there are two types of input shapes that can be used when creating a network. The batch input shape, and the normal input shape. 
        These are multidimensional arrays that hold the shape of the incoming data that the network should handle, in this case images. 
        In keras running a tensor flow backened I had to change the default dimenson ordering to the dimenson ordering used for thaneo. 
        To set this the following 2 lines of code are needed:
        <br>
        <center style="font-family:courier;">from keras import backend as K</center>
        <br>
        <center style="font-family:courier;">K.set_image_dim_ordering('th')</center>
<br>
    then the input shape should be set as <span style="font-family:courier;">(batch_size, depth, width, height)</span>, or you can omit the batch_size and set it as 
    <span style="font-family:courier;">(depth, width, height)</span> and the batch size will default to 1
<br>
    the convolution layers should be set as <span style="font-family:courier;">(num of filters, window_width, window_height)</span>
    </p>
</div>

<div>
    <H3 style="color:#000000;">Creating a simple classifyer</H3>
    <p style="text-indent:2.0em;">Here is a link to the code 
	<a href="https://github.com/IsaacIsOkay/learnerWorkspace/blob/master/dogsAndCats.py" target="_blank">link</a> 
	this creates a simple, 2 class, classifier that classifies between cat and dog images. 
	You can run the predict function from keras to test it on an image, and then the prediction returned is an array
        that contains how much keras thinks the image fits its model for a cat and a dog. This can only be done after 
	the network is trained (line 84 in the code).
    </p>
</div>

<div>
    <H3 style="color:#000000;">Pulling in training data</H3>
    <p style="text-indent:2.0em;">
        The way that I was able to pull in my data was through the use of the data generator
        and the fit generator provided in keras. To use this you must
	Split you different images into seperate directories bassed on their classes. Then you can load in you data with the image generator. This uses the 
	<span style="font-family:courier;"> flow_from_directory</span> from the preprocessing module. The dogs and cats
	classifier also shows the basics of how to use this functionality (line 59-66 in the code).
        However there are otherways to pull in image data using the Image handelers provided by keras.
        the image handeling is contained in:
	<br>
	<center style="font-family:courier;">keras.preprocessing.image</center>
	<br>
	the <span style="font-family:courier;">load_img</span> module from this package allows you
	to load in images into a PIL format that is a numpy array formated in a way that
	you can send into your convolution network.
	</p>
</div>

<div>
    <H3 style="color:#000000;">Dealing with small amounts of data</H3>
    <p style="text-indent:2.0em;">
        If your data set is small keras provides some image augmentation tools so that
        you can greatly expand the amount of training data available.
        The <a href="https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html" target="_blank">image augmentation</a>
        blog has a great explanation, and example of how to do this in keras, and the following is my implementation of it
        <a href="https://github.com/IsaacIsOkay/learnerWorkspace/blob/master/dataAugmentation.py" target="_blank">link</a>. 
	I had to change a bit of code to get the example from the blog so if you have trouble running
        code from the blog post my code from the previouse link might help.
	<br>
	running the code I get some of the following augmented images
	<br>
	<br>
	<table>
		<tr>
			<td><img src="https://github.com/IsaacIsOkay/learnerWorkspace/blob/master/preview/cat_0_1623.jpeg?raw=true" style="width:150px;height:100px;"></td>
			<td><img src="https://github.com/IsaacIsOkay/learnerWorkspace/blob/master/preview/cat_0_2722.jpeg?raw=true" style="width:150px;height:100px;"></td>
			<td><img src="https://github.com/IsaacIsOkay/learnerWorkspace/blob/master/preview/cat_0_2957.jpeg?raw=true" style="width:150px;height:100px;"></td>
		</tr>
		<tr>
			<td><img src="https://github.com/IsaacIsOkay/learnerWorkspace/blob/master/preview/cat_0_3051.jpeg?raw=true" style="width:150px;height:100px;"></td>
			<td><img src="https://github.com/IsaacIsOkay/learnerWorkspace/blob/master/preview/cat_0_8084.jpeg?raw=true" style="width:150px;height:100px;"></td>
			<td><img src="https://github.com/IsaacIsOkay/learnerWorkspace/blob/master/preview/cat_0_3544.jpeg?raw=true" style="width:150px;height:100px;"></td>
		</tr>
		<tr>
			<td><img src="https://github.com/IsaacIsOkay/learnerWorkspace/blob/master/preview/cat_0_9404.jpeg?raw=true" style="width:150px;height:100px;"></td>
			<td><img src="https://github.com/IsaacIsOkay/learnerWorkspace/blob/master/preview/cat_0_9715.jpeg?raw=true" style="width:150px;height:100px;"></td>
			<td><img src="https://github.com/IsaacIsOkay/learnerWorkspace/blob/master/preview/cat_0_3888.jpeg?raw=true" style="width:150px;height:100px;"></td>
		</tr>
	</table>
		
    </p>
    
</div>

<div>
    <H3>Seeing What the network Sees</H3>
    <p style="text-indent:2.0em;"> The keras blog has an exaple of how to see what the filters that
	are being generated by the network look like. I don't entirely understand how this works, but
	it is cool to look at. <a href="https://github.com/IsaacIsOkay/learnerWorkspace/blob/master/exploreLayers.py">This</a> is my implementation for the dogs and cats classifier. Most of 
	the feature maps are just static because there are only 2 different classes, and the network
	is not very deep, however it could be implemented on larger networks giving more interesting
	resutls.
    </p>
</div>

<div>
    <H3 style="color:#000000;">Fine tuning and loading weights</H3>
    <p style="text-indent:2.0em;">
        Often it is benificial to fine tune a network instead of training one
        from scratch. This can be easily done in keras where you can load weights and 
        freeze layers of already trained networks. Keras has functions that allow you to easily
        load in most of the popular, image net trained, neural networks. These are great for
        fine tuning becasue you can just chop of the the regression head used for classification
        by simply changing the argument 'include_top' to false. Then you can add your own regression 
	head freezed the feature maps from the top layers and then train a regression head to 
	classify your data. You can also chop out some of the feature maps from the higher layers 
	and train your own in their place. This is especially useful when you data is very 
	different from what image net contains because most of the more primative feature maps 
	will train simillaryly with your data.
	<br>
	To do this in keras there are tools for commonly used pretrained convolution networks (such as the VGG network, Alex Net, googleNet, ect.) They can be loaded with the applications
	module in keras for the VGG network an example of loading this without the regression head would be using the following code:
	<center style="font-family:courier;">model = applications.VGG16(include_top=False, weights='imagenet')</center>
	If you want to load in weights for your own network you can also do this. After training your own data, and saving it as an h5py file you rebuild the network with the sequential model with the number 	of layers you want to load from your originally trained network. Then you can set the weights for this model with the load_weights function
		<center style="font-family:courier;">model.load_weights(weights_path, by_name=True)</center>
     </p>
</div>

</body>
</div>
</html>

