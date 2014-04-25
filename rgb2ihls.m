%----------------------
%Andrew Mendez
% 4/25/14
%Currently trying to convert code from : http://www.shatterline.com/SkinDetection.html
% to matlab function that converts rgb to ihls
%------------------

%constants
pi = 3.142;

matrix = [.0000,  0.7875,  0.3714;1.0000, -0.2125, -0.2059;1.0000, -0.2125,  0.9488];

%get the inverse of the matrix
invMat = inv(matrix);

%taken from, A 3D-polar Coordinate Colour Representation Suitable for Image Analysis
%by Allan Hanbury and Jean Serra
function[r,g,b] = rgb2ihls(r,g,b)  
	% Here be the algorithm
	
% 	#luminance
% 	#y=0.2125*r + 0.7154*g + 0.0721*b;
	y = 0.299 * r + 0.587 * g + 0.114 * b;
% 	#y=(r + g + b)/3.0;
% 	#y=y*255;
	
% 	#hue
	C1 = r - 0.5*g - 0.5*b
	C2 = -sqrt(3.0)/2.0*g + sqrt(3.0)/2.0*b
	C = sqrt(C1^2 + C2^2)
	
	if (C ~=0 && C2 <= 0) 
		h <- acos(C1 / C)
	
	else if (C ~=0 && C2 > 0) 
		h = 2.0 * pi - acos(C1 / C) 
	
	else 
% 		#warning("C is equal to 0 in the RGB to IHLS transform!")
		h = 0
% 		#using h <- 0 when C <- 0
% 		#indic=find(C == 0); h(indic)=0;
	
	
% 	#convert h to degrees
% 	# do we need this step for classsification purposes
	h = h/pi;
	h = h*180.0;
	
% 	#saturation		
% 	#k=h./(pi/3.0);
% 	#k=floor(k);
% 	#hstar=h - k.*(pi/3.0);		
% 	#s=(2.0*C.*sin(-hstar + (2.0*pi/3.0)))/sqrt(3.0);
	
% 	#simpler saturation expression which produces the same as above
	s = max(r,max(g,b)) - min(r,min(g,b))
	
	return (h,y,s)
     end

ihls2rgb <- function(h,y,s) {
	
% 	#convert h to radians
	h <- h / 180.0
	h <- h * pi
	
% 	#chroma C
	k <- h / (pi/3.0)
	k <- floor(k)
	hstar <- h - k * (pi/3.0)
	Ct <- (sqrt(3.0) * s)
	Cb <- (2.0 * sin(-hstar + (2.0*pi/3.0)))
	C <- Ct/Cb
	
% 	#C1 and C2
	C1 <- C * cos(h);
	C2 <- -C * sin(h);
	
	
	rgb <- inv.matrix %*% c(y, C1, C2) 
	r <- rgb[1]
	g <- rgb[2]
	b <- rgb[3]
	#sort out rounding errors
	if (r > 1) r <- 1
	if (g > 1) g <- 1
	if (b > 1) b <- 1
	
	if (r < 0) r <- 0
	if (g < 0) g <- 0
	if (b < 0) b <- 0
	
	rgb[1] <- r
	rgb[2] <- g
	rgb[3] <- b
	
	return (rgb) 
}


diff.image <- function(img.matrix, mask.matrix.binary) {
	for (i in seq(1:nrow(img.matrix))) {
		for (j in seq(1:ncol(img.matrix))){
			if (mask.matrix.binary[i,j] == "N") { #not skin
				img.matrix[i,j,] <- c(0,0,0)	
			} 
		}
	}
	return(img.matrix)
} 

# the truth matriux is a 2-D character array 
mask.image <- function(rgb.matrix, truth.matrix) {
	#print("Truth Dim:"); print(dim(truth.matrix))
	#print("RGB Dim:"); print(dim(rgb.matrix))
	for (i in 1:nrow(rgb.matrix)) {
		for (j in 1:ncol(rgb.matrix)){
			if(truth.matrix[i,j] == 'N'){
				rgb.matrix[i,j,1] <- 0
				rgb.matrix[i,j,2] <- 0
				rgb.matrix[i,j,3] <- 0
			}
		}
	}
	return(rgb.matrix)
}



#use lapply here
image.rgb2ihls <- function(img.matrix, ihls.matrix) {
	for (i in seq(1:nrow(img.matrix))) {
		for (j in seq(1:ncol(img.matrix))){
			#ihls.point <- rgb2ihls(img.matrix[i,j,1], img.matrix[i,j,2], img.matrix[i,j,3])
			
			ihls.matrix[i,j,] <- rgb2ihls(img.matrix[i,j,1], img.matrix[i,j,2], img.matrix[i,j,3])
		}
	}
	return (ihls.matrix)
}

image.ihls2rgb <- function(ihls.matrix, rgb.prime.matrix) {
	for (i in seq(1:nrow(ihls.matrix))) {
		for (j in seq(1:ncol(ihls.matrix))){
			
			rgb.prime.matrix[i,j,] <- ihls2rgb(ihls.matrix[i,j,1], ihls.matrix[i,j,2], ihls.matrix[i,j,3])
		}
	}
	return(rgb.prime.matrix)
}


image.binarize <- function(img.matrix.groundtruth, img.matrix.binary) {
	print(nrow(img.matrix.groundtruth)); print(ncol(img.matrix.groundtruth));
	print(dim(img.matrix.groundtruth))
	print(dim(img.matrix.binary))
	for (i in seq(1:nrow(img.matrix.groundtruth))) {
		for (j in seq(1:ncol(img.matrix.groundtruth))){
			#print(img.matrix.groundtruth[i,j,1])
			if(img.matrix.groundtruth[i,j,1] > 0.1){
				img.matrix.binary[i,j] <- 'S'
			}
			else 
				img.matrix.binary[i,j] <- 'N'
		}
	}
	return(img.matrix.binary)
}

# 50 by 50 square with 20 by 20 red square in the middle
synthetic.create.redsquare <- function(rgb.matrix, inner) {
	
	# initialize everything to white
	for (i in 1:nrow(rgb.matrix)) {
		for (j in 1:ncol(rgb.matrix)){
			rgb.matrix[i,j,1] <- 0
			rgb.matrix[i,j,2] <- 0
			rgb.matrix[i,j,3] <- 0
		}
	}
	
	# initialize middlw 20 pixels to red
	for (i in ((1+inner):(nrow(rgb.matrix)-inner))) {
		for (j in ((1+inner):(ncol(rgb.matrix)-inner))){
			rgb.matrix[i,j,1] <- 1
			rgb.matrix[i,j,2] <- 0
			rgb.matrix[i,j,3] <- 0
		}
	}
	
	return(rgb.matrix)
}

#for synthetic images
synthetic.binarize <- function(img.matrix.groundtruth, inner) {

	# initialize everything to non-skin
	for (i in 1:nrow(img.matrix.groundtruth)) {
		for (j in 1:ncol(img.matrix.groundtruth)){
			img.matrix.groundtruth[i,j] <- 'N'
		}
	}
	
	# initialize middle pixels skin, take innrer as parameter
	for (i in ((1+inner):(nrow(img.matrix.groundtruth)-inner))) {
		for (j in (1+inner):(ncol(img.matrix.groundtruth)-inner)){
			img.matrix.groundtruth[i,j] <- 'S'
		}
	}	
	return(img.matrix.groundtruth)
}



plot.gt <- function(matrix.gt) {
	
	m2 <- matrix.gt
	m2[] <- c("red", "blue")[match(matrix.gt, c("N","S"))]
	m2
	plot(row(m2), col(m2), col=m2, pch=18, cex=4) #prints diamonds
	
}


convert.matrix.2.attribute.table <- function(matrix, table) {
	k <- 1
	for (i in 1:nrow(matrix)) {
		for (j in 1:ncol(matrix)){
			#print (i); print(j); print(k);
			#print (matrix[i,j,1]); print (matrix[i,j,2]); print (matrix[i,j,3]); 
			table[k,1] <- matrix[i,j,1]
			table[k,2] <- matrix[i,j,2]
			table[k,3] <- matrix[i,j,3]
			
			k <- k+1
		}
	}
	#print(table)
	return(table)
}


convert.truth.table.2.matrix <- function(table, matrix) {
	k <- 1
	for (i in 1:nrow(matrix)) {
		for (j in 1:ncol(matrix)){
			matrix[i,j] <- table[k] 			
			k <- k+1
		}
	}
	#print(matrix)
	return(matrix)
}

convert.matrix.2.truth.table <- function (matrix, table) {
	
	k <- 1
	for (i in 1:nrow(matrix)) {
		for (j in 1:ncol(matrix)){
			table [k] <- matrix[i,j] 			
			k <- k+1
		}
	}
	#print(table)
	return(table)
	
}