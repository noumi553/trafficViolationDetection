from Module.red_and_green import main
from Module.speed import SpeedDetectorMain
from Module.helmatdetection import Main
from Module.wrongsideviolationdetection import WrongWayViolation

def FunctionMain():
    main()
    Object = SpeedDetectorMain
    Object.main()
    Main()
    Objects = WrongWayViolation
    Objects.Main()

FunctionMain()