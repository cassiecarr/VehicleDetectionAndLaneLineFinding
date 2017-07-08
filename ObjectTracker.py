import numpy as np
class Cars:
    def __init__(self):
        self.car_count = 0

        self.these_cars = []

        self.all_cars = []

        self.car_average_boxes = []

    def cars_update(self,boxes):
        self.these_cars = boxes
        
        print("self.these_cars")
        print(self.these_cars)

        # Add empty array for every new car found
        if len(boxes) > self.car_count:
            for i in range(self.car_count, len(boxes)):
                self.all_cars.append([])
                self.car_count += 1
        # for i in range(0, self.car_count):
        #     if((i < len(boxes)) & (len(self.all_cars[0]) > 0)):
        #         if(abs(boxes[i][0][0] - self.all_cars[i][0][0][0]) > 50):
        #             if(i == (len(self.all_cars)-1)):
        #                 self.all_cars.append([])
        #                 self.car_count += 1
    
        for i in range(0, self.car_count):

            if ((i < len(boxes)) & (len(boxes) > 0)):
                # if((i < len(self.all_cars)) & (len(self.all_cars[0]) > 0)):
                #     if (len(self.all_cars[i]) > 0):
                #         if((abs(boxes[i][0][0] - self.all_cars[i][0][0][0]) > 50)):
                #             print("*")
                #             boxes.append(boxes[i])
                #             continue

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
            if(len(self.all_cars[i]) > 1):
                avg_boxes = np.mean(np.array(self.all_cars[i]), axis=0).astype(int)
                self.car_average_boxes.append(avg_boxes)

        print("self.all_cars")
        print(self.all_cars)


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
        print("self.car_average_boxes")
        print(self.cars.car_average_boxes)

    def get_boxes(self):

        return self.cars.car_average_boxes

    def get_poly_left(self):

        return self.poly_left_average

    def get_poly_right(self):

        return self.poly_right_average
