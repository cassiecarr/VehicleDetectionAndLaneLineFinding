import numpy as np
class Cars:
    def __init__(self):
        self.car_count = 0

        self.these_cars = []

        self.all_cars = []

        self.car_average_boxes = []

    def cars_update(self,boxes):
        self.these_cars = boxes
 
        # Remove empty array and reduce car count
        if(len(self.all_cars) > 0):
            if (([]) in self.all_cars):
                self.all_cars.remove([])
                self.car_count -= 1

        # Add empty array for every new car found
        if len(boxes) > self.car_count:
            for i in range(self.car_count, len(boxes)):
                self.all_cars.append([])
                self.car_count += 1
        
        # Loop through all the cars
        for i in range(0, self.car_count):
            
            # If there are boxes for this car count
            if ((i < len(boxes)) & (len(boxes) > 0)):

                # Check if boxes are close to all_cars array at car count
                if((i < len(self.all_cars)) & (len(self.all_cars[0]) > 0)):
                    if(len(self.all_cars[i]) > 0):
                        if((abs(boxes[i][0][0] - self.all_cars[i][0][0][0]) > 50)):
                            # If boxes are far from all_cars array at car count:
                            # Add to new array if last arry in all_cars
                            # If not last array, remove current item at all_cars and then check next car
                            if(i == (len(self.all_cars)-1)):
                                self.all_cars.append([])
                                self.car_count += 1
                                self.all_cars[i+1].append(boxes[i])
                                continue
                            else:
                                self.all_cars[i].pop(0)
                                boxes.append(boxes[i])
                                continue

                if(len(self.all_cars[i]) < 10):
                    self.all_cars[i].append(boxes[i])
                
                else:
                    self.all_cars[i].pop(0)
                    self.all_cars[i].append(boxes[i])
            else:
                if(len(self.all_cars[i]) > 0):
                    self.all_cars[i].pop(0)

        self.car_average_boxes = []
        for i in range(0, len(self.all_cars)):
            if(len(self.all_cars[i]) > 3):
                avg_boxes = np.mean(np.array(self.all_cars[i]), axis=0).astype(int)
                self.car_average_boxes.append(avg_boxes)


class Object:
    def __init__(self):
        self.count = 0
        
        self.this_poly_left = []        # Polynomial coefficients for left lane in this frame
        self.this_poly_right = []       # Polynomial coefficients for right lane in this frame
        
        self.all_poly_left = []         # Polynomial coefficients for left lane in the last 10 frames
        self.all_poly_right = []        # Polynomial coefficients for right lane in the last 10 frames
        
        self.poly_left_average = []     # Average polynomial coefficients for left lane
        self.poly_right_average = []    # Average polynomial coefficients for right lane

        self.cars = Cars()

    def update(self,boxes,poly_left,poly_right):
        self.this_poly_left = poly_left
        self.this_poly_right = poly_right

        if self.count < 10:
            self.all_poly_left.append(poly_left)
            self.all_poly_right.append(poly_right)
            self.count += 1
        else:
            self.all_poly_left.pop(0)
            self.all_poly_right.pop(0)
            self.all_poly_left.append(poly_left)
            self.all_poly_right.append(poly_right)
        
        self.poly_left_average = np.mean(np.array(self.all_poly_left), axis=0)
        self.poly_right_average = np.mean(np.array(self.all_poly_right), axis=0)

        self.cars.cars_update(boxes)

    def get_boxes(self):

        return self.cars.car_average_boxes

    def get_poly_left(self):

        return self.poly_left_average

    def get_poly_right(self):

        return self.poly_right_average
