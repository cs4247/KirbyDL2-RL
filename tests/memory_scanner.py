from pyboy import PyBoy
from KirbyGame import KirbysDreamland2
from threading import Thread
import time

pyboy = KirbysDreamland2('roms/kirby2.gb', debug = True)
print(f"Playing {pyboy.cartridge_title()}")
pyboy.enable_invincibility()
addresses = range(0x0000, 0xFFFF)
last_values = [0]*0xFFFF

while not pyboy.tick():
    if pyboy.paused:
        print(pyboy)
        print(
            "Enter task:\n"
            "  s: Search for a specific value in memory.\n"
            "  nz: Search for non-zero values in memory.\n"
            "  c: Search for memory addresses where the value has changed.\n"
            "  u: Search for memory addresses where the value has remained the same.\n"
            "  i: Search for memory addresses where the value has increased.\n"
            "  d: Search for memory addresses where the value has decreased.\n"
            "  r: Read the value at a specific memory address.\n"
            "  ra: Read every address currently in search.\n"
            "  n: Do nothing, unpause.\n"
            "  set: Set memory address value.\n"
            "  track: Print addresses while running.\n"
            "  st: Stop printing addresses.\n"
            "  rs: Restart the search, resetting all addresses."
        )

        mode = input("Your choice: ")
        if mode == 's':
            target = int(input('Enter target: '))
            
            addresses = [a for a in addresses if pyboy.get_memory_value(a) == target]
            print(', '.join(hex(address) for address in addresses))
        elif mode == 'nz':
            print('Searching none-zero')
            addresses = [a for a in addresses if pyboy.get_memory_value(a) != 0]
            print(', '.join(hex(address) for address in addresses))
        elif mode == 'c':
            print('Changed addresses:')
            addresses = [a for a in addresses if pyboy.get_memory_value(a) != last_values[a]]
            print(', '.join(hex(address) for address in addresses))
        elif mode == 'u':
            print('Unchanged addresses.')
            addresses = [a for a in addresses if pyboy.get_memory_value(a) == last_values[a]]
            print(', '.join(hex(address) for address in addresses))
        elif mode == 'i':
            print('Increased addresses.')
            addresses = [a for a in addresses if pyboy.get_memory_value(a) > last_values[a]]
            print(', '.join(hex(address) for address in addresses))
        elif mode =='d':
            print('Decreased addresses.')
            addresses = [a for a in addresses if pyboy.get_memory_value(a) < last_values[a]]
            print(', '.join(hex(address) for address in addresses))
        elif mode == 'r':
            target = input('Enter addresses (comma separated): ')
            addresses = target.split(', ')
            for address in addresses:
                try:
                    a = int(address, 16)
                    value = pyboy.get_memory_value(a)
                    print(f"Value at {address}: {value}")
                except ValueError:
                    print(f"Invalid address: {address}")
        elif mode == 'ra':
            print(f"Current addresses: {', '.join(hex(address) for address in addresses)}")
            for a in addresses:
                print(f"{hex(a)}: {pyboy.get_memory_value(a)}")
        elif mode == 'n':
            print('Unpausing in 1 second')
            time.sleep(1)
        elif mode == 'rs':
            print('Restarting search.')
            addresses = range(0x0000, 0xFFFF)
        elif mode == 'set':
            target = input('Enter address: ')
            a = int(target, 16)
            value = int(input("Enter value: "))
            pyboy.set_memory_value(a, value)
        elif mode == 'st':
            tracked_addresses = []
        else:
            print('Bad input')

        for a in range(0x0000, 0xFFFF):
            last_values[a] = pyboy.get_memory_value(a)
        pyboy._unpause()

pyboy.stop()
