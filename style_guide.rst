===========
Style guide 
===========

In general, Covasim follows Google's `style guide <https://google.github.io/styleguide/pyguide.html>`_. If you simply follow that, you can't go too wrong. However, there are a few "house style" differences, which are described here.

Covasim uses ``pylint`` to ensure style conventions. To check if your styles are compliant, run ``./tests/check_style``.



Design philosophy
=================

Covasim's overall design philosophy is "**Common tasks should be simple**, while uncommon tasks can't always be simple, but still should be possible".


Writing for the right audience
------------------------------

The audience for Covasim is *scientists*, not software developers. Assume that the average Covasim user dislikes coding and wants something that *just works*. Implications of this include:

- Commands should be short, simple, and obvious, e.g. ``cv.Sim().run().plot()``.
- Be as flexible as possible with user inputs. If a user could only mean one thing, do that. If the user provides ``[0, 7, 14]`` but the function needs an array instead of a list, convert the list to an array automatically (``sc.toarray()`` exists for exactly this reason).
- If there's a "sensible" default value for something, use it. Users shouldn't *have* to think about false positivity rate or influenza-like illness prevalence if they just want to quickly add testing to their simulation via ``cv.test_prob(0.1)``.
- However, hard-code as little as possible. For example, rather than defining a variable at the top of the function, make it a keyword argument with a default value so the user can modify it if needed. As another example, pass keyword arguments where possible -- e.g., ``to_json(..., **kwargs)`` can pass arbitrary extra arguments via ``json.dump(**kwargs)``.
- Ensure the logic, especially the scientific logic, of the code is as clear as possible. If something is "bad" coding style but good science, you should probably do it anyway. 


Workload considerations
-----------------------

The total work your code creates is:

.. math::

    W = \sum_p \left( u_p + n_p \times r_p + m_p \times e_p \right)

where:

- *W* is the total work
- *p* is each person
- *u* is the ramp-up time
- *n* is the number of reads
- *r* is the time per read
- *m* is the number of edits
- *e* is the time per edit

Implications of this include:

- Common mistakes are to overemphasize *p* = 0 (you) over *p* > 0, and *e* (edit time) over *u* (ramp-up time) and *r* (read time). 
- Assume people of different backgrounds and skill levels will be using/interacting with this code. You might be comfortable with lambda functions and overriding dunder methods, but assume others are not. Use these "advanced features" only as a last resort.
- Similarly, try to avoid complex dependencies (e.g. nested class inheritance) as they increase ramp-up time, and make it more likely something will break. (But equally, don't repeat yourself -- it's a tradeoff.)
- Err on the side of more comments, including line comments. Logic that is clear to you now might not be clear to anyone else (or yourself 3 months from now). If you use a number that came from a scientific paper, please for the love of all that is precious put a link to that paper in a comment.



House style
===========

As noted above, Covasim follows Google's style guide (GSG), with these exceptions (numbers refer to Google's style guide):



2.8 Default Iterators and Operators (`GSG28 <https://google.github.io/styleguide/pyguide.html#28-default-iterators-and-operators>`_)
------------------------------------------------------------------------------------------------------------------------------------

**Difference**: It's fine to use ``for key in obj.keys(): ...`` instead of ``for key in obj: ...``.

**Reason**: In larger functions with complex data types, it's not immediately obvious what type an object is. While ``for key in obj: ...`` is fine, especially if it's clear that ``obj`` is a dict, ``for key in obj.keys(): ...`` is also acceptable if it improves clarity.



2.21 Type Annotated Code (`GSG221 <https://google.github.io/styleguide/pyguide.html#221-type-annotated-code>`_)
---------------------------------------------------------------------------------------------------------------

**Difference**: Do *not* use type annotations.

**Reason**: Type annotations are useful for ensuring that simple functions do exactly what they're supposed to as part of a complex whole. They prioritize consistency over convenience, which is the correct priority for low-level library functions, but not for functions and classes that aim to make it as easy as possible for the user. 

For example, in Covasim, dates can be specified as numbers (``22``), strings (``'2022-02-02'``), or date objects (``datetime.date(2022, 2, 2)``), etc. Likewise, many quantities can be specified as a scalar, list, or array. If a function *usually* only needs a single input but can optionally operate on more than one, it adds burden on the user to require them to provide e.g. ``np.array([0.3])`` rather than just ``0.3``. In addition, most functions have default arguments of ``None``, in which case Covasim will use "sensible" defaults.

Attempting to apply type annotations to the flexibility Covasim gives to the user would result in monstrosities like:

.. code-block:: python

    def count_days(self, start_day: typing.Union[None, str, int, dt.date, dt.datetime],
                   end_day: typing.Union[None, str, int, dt.date, dt.datetime]) -> int:
        return self.day(end_day) - self.day(start_day)

If your function is written in such a way that type definitions would be helpful, consider if there is a way to rewrite it such that (a) it can accept a wider range of inputs, and/or (b) you can make it clearer what is allowed. For example, ``values`` should likely accept a list or array of any numeric type; ``label`` should be a single string; ``labels`` should be a list of strings.

Note that you *can* (and should) use type annotations in your docstrings. For example, the above method could be written as:

.. code-block:: python

    def count_days(self, start_day, end_day):
        """ Count days between start and end relative to "sim time"

        Args:
            start_day (int/str/date): The day to start counting
            end_day   (int/str/date): The day to stop counting

        Returns:
            Number of days elapsed

        **Example**::
        
            sim.count_days(45, '2022-02-02')
        """
        return self.day(end_day) - self.day(start_day)



3.2 Line length (`GSG32 <https://google.github.io/styleguide/pyguide.html#32-line-length>`_)
--------------------------------------------------------------------------------------------

**Difference**: Long lines are not *great*, but are justified in some circumstances.

**Reason**: Line lengths of 80 characters are due to `historical limitations <https://en.wikipedia.org/wiki/Characters_per_line>`_. Think of lines >80 characters as bad, but breaking a line as being equally bad. Decide whether a long line would be better implemented some other way -- for example, rather than breaking a >80 character list comprehension over multiple lines, use a ``for`` loop instead. Always keep literal strings together (do not use implicit string concatenation).

Line comments are encouraged in Covasim, and these can be as long as needed; they should not be broken over multiple lines to avoid breaking the flow of the code. A 50-character line with a 150 character line comment after it is completely fine. The rationale is that long line comments only need to be read very occasionally; if they are broken up over multiple lines, then they have to be scrolled past *every single time*. Since scrolling vertically is such a common task, it is important to minimize the amount of effort required (i.e., minimizing lines) while not sacrificing clarity. Vertically compact code also means more will fit on your screen (and thence your brain).

Examples:

.. code-block:: python

    # Yes: it's a bit longer than 80 chars but not too bad
    foo_bar(self, width, height, color='black', design=None, x='foo', emphasis=None)

    # No: the cost of breaking the line is too high
    foo_bar(self, width, height, color='black', design=None, x='foo',
            emphasis=None)

    # No: line is needlessly long, rename variables to be more concise to avoid the need to break
    foo_bar(self, object_width, object_height, text_color='black', text_design=None, x='foo', text_emphasis=None)

    # No: line is too long
    foo_bar(self, width, height, design=None, x='foo', emphasis=None, fg_color='black', bg_color='white', frame_color='orange')

    # Yes: if you do need to break a line, try to break at a semantically meaningful point
    foo_bar(self, width, height, design=None, x='foo', emphasis=None,
            fg_color='black', bg_color='white', frame_color='orange')

    # Yes: long line comments are ok
    foo_bar(self, width, height, color='black', design=None, x='foo') # Note the difference with bar_foo(), which does not perform the opposite operation



3.5 Blank Lines (`GSG35 <https://google.github.io/styleguide/pyguide.html#35-blank-lines>`_)
--------------------------------------------------------------------------------------------

**Difference**: Always use (at least) one extra blank line between levels as within a level.

**Reason**: Google's recommendation (two blank lines between functions or classes, one blank line between methods) is appropriate for small-to-medium classes and methods. However, for large methods (e.g. >50 lines) with multiple blank lines within them, using only a single blank line can mark it hard to tell where one method stops and the next one starts. Thus, for a method that contains blank lines within itself, use *two* blank lines between methods (and then do that consistently for the rest of the class). For separating large classes/functions (>500 lines), or classes whose methods are separated by two blank lines, separating them by three blank lines is preferable.

While not explicitly covered by the Google style guide, **return** statements should be used at the end of each function and method, even if that block returns ``None`` (in which case use ``return``, not ``return None``). This helps delimit larger methods/functions. However, always ask whether a function/method *should* return ``None``. Following the pandas convention, many Covasim methods return ``self``, which is what enables "chaining" patterns such as ``cv.Sim().run().plot()``.



3.6 Whitespace (`GSG36 <https://google.github.io/styleguide/pyguide.html#36-whitespace>`_)
------------------------------------------------------------------------------------------

**Difference**: You *should* use spaces to vertically align tokens.

**Reason**: This convention, which is a type of `semantic indenting <https://gist.github.com/androidfred/66873faf9f0b76f595b5e3ea3537a97c>`_, can greatly increase readability of the code by drawing attention to the semantic similarities and differences between consecutive lines.

Consider how hard it is to debug this code:

.. code-block:: python

    # Perform updates
    self.init_flows()
    self.flows['new_infectious'] += self.check_infectious()
    self.flows['new_symptomatic'] += self.check_symptomatic()
    self.flows['new_severe'] += self.check_symptomatic()
    self.flows['new_critical'] += self.check_critical()
    self.flows['new_recoveries'] += self.check_recovery()

vs. this:

.. code-block:: python

    # Perform updates
    self.init_flows()
    self.flows['new_infectious']  += self.check_infectious()
    self.flows['new_symptomatic'] += self.check_symptomatic()
    self.flows['new_severe']      += self.check_symptomatic()
    self.flows['new_critical']    += self.check_critical()
    self.flows['new_recoveries']  += self.check_recovery()

In the second case, the typo (repeated ``check_symptomatic()``)  immediately jumps out, whereas in the first case, it requires careful scanning of each line.

Vertically aligned code blocks also make it easier to edit code using editors that allow multiline editing (e.g., `Sublime <https://www.sublimetext.com/>`_). However, use your judgement -- there are cases where it does more harm than good, especially if the block is small, or if egregious amounts of whitespace would need to be used to achieve alignment:

.. code-block:: python

    # Yes
    test_prob  = 0.1 # Per-day testing probability
    vax_prob   = 0.3 # Per-campaign vaccination probability
    trace_prob = 0.8 # Per-contact probability of being traced

    # Yes
    t = 0 # Start day
    omicron_vax_prob = dict(low=0.05, high=0.1) # Per-day probability of receiving Omicron vaccine

    # Hell no
    t                = 0                        # Start day
    omicron_vax_prob = dict(low=0.05, high=0.1) # Per-day probability of receiving Omicron vaccine



3.8.5 Block and Inline Comments (`GSG385 <https://google.github.io/styleguide/pyguide.html#385-block-and-inline-comments>`_)
----------------------------------------------------------------------------------------------------------------------------

**Difference**: Use either one or two spaces between code and a line comment.

**Reason**: The advice "Use two spaces to improve readability" dates back to the era when most code was viewed as plain text. Now that virtually all editors have syntax highlighting, it's no longer really necessary. There's nothing *wrong* with two spaces, but if it's easier to type one space, do it.



3.10 Strings (`GSG310 <https://google.github.io/styleguide/pyguide.html#310-strings>`_)
---------------------------------------------------------------------------------------

**Difference**: Always use f-strings or addition.

**Reason**: It's just nicer. Compared to ``'{}, {}'.format(first, second)`` or ``'%s, %s' % (first, second)``, ``f'{first}, {second}'`` is both shorter and clearer to read. However, use concatenation if it's simpler, e.g. ``third = first + second`` rather than ``third = f'{first}{second}'`` (because again, it's shorter and clearer).



3.13 Imports formatting (`GSG313 <https://google.github.io/styleguide/pyguide.html#313-imports-formatting>`_)
-------------------------------------------------------------------------------------------------------------

**Difference**: Group imports logically rather than alphabetically.

**Reason**: Covasim modules shouldn't need a long list of imports. Sort imports as in Google's style guide, but second-order sorting should be grouped by "level", e.g. low-level libraries first (e.g. file I/O), then high-level libraries last (e.g., plotting). For example:

.. code-block:: python

    import os
    import shutil
    import numpy as np
    import pandas as pd
    import pylab as pl
    import seaborn as sns
    from .covasim import defaults as cvd
    from .covasim import plotting as cvpl

Note the logical groupings -- standard library imports first, then numeric libraries, with Numpy coming before pandas since it's lower level; then external plotting libraries; and finally internal imports.

Note also the use of ``import pylab as pl`` instead of the more common ``import matplotlib.pyplot as plt``. These are functionally identical; the former is used simply because it is easier to type, but this convention may change to the more standard Matplotlib import in future.


3.14 Statements (`GSG314 <https://google.github.io/styleguide/pyguide.html#314-statements>`_)
---------------------------------------------------------------------------------------------

**Difference**: Multiline statements are *sometimes* OK.

**Reason**: Like with semantic indenting, sometimes it causes additional work to break up a simple block of logic vertically. However, use your judgement, and err on the side of Google's style guide. For example:

.. code-block:: python

    # Yes
    if foo: bar(foo)

    # Yes
    if foo:
        bar(foo)
    else:
        baz(foo)

    # Borderline
    if foo: bar(foo)
    else:   baz(foo)

    # Yes, but maybe rethink your life choices
    if   foo == 0: bar(foo)
    elif foo == 1: baz(foo)
    elif foo == 2: bat(foo)
    elif foo == 3: bam(foo)
    elif foo == 4: bak(foo)
    else:          zzz(foo)

    # No: definitely rethink your life choices
    if foo == 0:
        bar(foo)
    elif foo == 1:
        baz(foo)
    elif foo == 2:
        bat(foo)
    elif foo == 3:
        bam(foo)
    elif foo == 4:
        bak(foo)
    else:
        zzz(foo)

    # OK
    try:
        bar(foo)
    except:
        pass

    # Also OK
    try:    bar(foo)
    except: pass

    # No: too much whitespace and logic too hidden
    try:               bar(foo)
    except ValueError: baz(foo)



3.16 Naming (`GSG316 <https://google.github.io/styleguide/pyguide.html#316-naming>`_)
-------------------------------------------------------------------------------------

**Difference**: Names should be consistent with other libraries and with how the user interacts with the code.

**Reason**: Covasim interacts with other libraries, especially Numpy and Matplotlib, and should not redefine these libraries' names. For example, Google naming convention would prefer ``fig_size`` to ``figsize``, but Matplotlib uses ``figsize``, so this should also be the name preferred by Covasim. (This applies if the variable name is *only* used by source libraries. If it's used by both, e.g. ``start_day`` used both directly by Covasim and by ``sc.date()``, it's OK to use the Google style convention.)

If an object is technically a class but is used more like a function (e.g. ``cv.change_beta()``), it should be named as if it were a function. A class is "used like a function" if the user is not expected to interact with it after creation, as is the case with most interventions. Thus ``cv.BaseVaccinate`` is a class that is intended to be used *as a class* (primarily for subclassing). ``cv.vaccinate_prob()`` is also a class, but intended to be used like a function; ``cv.vaccinate()`` is a function which returns an instance of ``cv.vaccinate_prob`` or ``cv.vaccinate_num``. Because ``cv.vaccinate()`` and ``cv.vaccinate_prob()`` can be used interchangeably, they are named according to the same convention.

Names should be as short as they can be while being *memorable*. This is slightly less strict than being unambiguous. Think of it as: the meaning might not be clear solely from the variable name, but should be clear from the docstring and/or line comment, and from *that* point should be unambiguous. For example:

.. code-block:: python

    # Yes
    vax_prob = 0.3 # Per-campaign vaccination probability

    # Also OK (but be consistent!)
    vx_prob = 0.3 # Per-campaign vaccination probability

    # No, too verbose; many more characters but not much more information
    vaccination_probability = 0.3

    # No, not enough information to figure out what this is
    vp = 0.3

Underscores in variable names are generally preferred, but there are exceptions (e.g. ``figsize`` mentioned above). Always ask whether part of a multi-part name is providing necessary clarity (and if it's not, omit it). For example, if an intervention called ``antigen_test()`` uses a single variable for probability, call that variable ``prob`` rather than ``test_prob``.


Parting words
-------------

If in doubt, ask! Slack, Teams, email, GitHub -- all work. And don't worry about getting it perfect; any differences in style will be reconciled during code review and merge.