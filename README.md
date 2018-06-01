This is the infrastructure package for point clouds and raster data processing in The Photogrammetry and Laser Scanning Lab in the Technion. 

The infrastucture is object oriented. There are three data-set object ( PointSet, PointSubSet and RasterData) which have a basic set of properties, Factoy objects and Property objects. 

The Property objects relate to the data-set, and are built by a Factory object. In each Factory there are several functions that build the same property, but with a different algorithm. 

There are also visualization classes via which visualization of the point cloud is possible. 
