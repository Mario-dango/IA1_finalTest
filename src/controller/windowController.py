# -*- coding: utf-8 -*-
##### LIBRERIAS Y DEPENDENCIAS NECESARIAS PARA LA CLASE #####
from app.src.controller.controlLong import ControlLongitudinal
from app.src.controller.controlTrans import ControlTransversal
from app.src.model.archivosExcel import PlanillaModel
from app.src.view.mainWindow import VentanaPrincipal
from app.src.view.widgets import FileManager, VentanaEmergenteExcel, ObraDeArteVentana, PaqueteEstructuralVentana
from PyQt5.QtCore import QObject
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
################################################################


## CLASE DEFINIDA DEL CONTROLADOR PRINCIPAL ##
class MainController(QObject):
    
    ## MÉTODO CONSTRUCTOR DE LA CLASE PRINCIPAL
    def __init__(self, mainWindow = None):
        super().__init__()
        ##  Si la ventana no es pasada cómo parametro, crear objeto ventana principal
        if mainWindow is None:
            self.ventanaPrincipal = VentanaPrincipal()
        else:
            self.ventanaPrincipal = mainWindow
        self.qFile = FileManager()                                  ##
        self.planillaManager = PlanillaModel()                      ##
        self.longitudinalController = ControlLongitudinal()    ##
        self.transversalController = ControlTransversal()      ##
        # self.ventanaArte = VentanaArte()                            ##
        # self.ventanaPaqueteEstructural = None                       ##
        self.ventanaDatos = VentanaEmergenteExcel()                 ##
        ## Llamado al método de configuración de inicio para la clase
        self.setup()