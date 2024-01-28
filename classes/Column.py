class Column:
  def __init__(self, name, my_name, characteristics_dict):
    
    self.name = name
    self.my_name = my_name
    self.characteristics_dict = characteristics_dict
    #obj_column.characteristic_list = [Characteristic() for i in range(6)]
    


  def print_column(self):
    print('\n' + 'column name=' + self.name +
          '\n' + 'column name=' + self.my_name +
          '\n' + 'column list=' , self.characteristics_dict ,
          '\n')
