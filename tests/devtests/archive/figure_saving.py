'''
Test figure saving
'''
import covasim as cv
import os

def make():

    print('Making figure')
    cv.Sim().run(do_plot=True)
    filename = cv.savefig(comments='good figure')

    return filename

filename = make()

print(f'Loading from {filename}:')
cv.get_png_metadata(filename)

os.remove(filename)
print('Done')


