# Ushichka data management plan
*Date of document initation: 2021-03-18*

*This document is still work in progress. The ideas here are rought*
The [Ushichka dataset](https://thejasvibr.github.io/ushichka/) has a variety
of data types, and will produce an even larger set of data types as work on 
it proceeds. What is the best tool to handle all the data types, and the 
relationships they have to each other. 

## Database management systems -- which one?
My first thought is to use a database management system (DBMS). DBMS's can store
multiple data types and the relationships between them. DBMS also seems to be
the most popular option I could find after a short online search. However, there
are a whole bunch of tools. Which one I use will depend on the final goals of 
the project itself. 

I have considerd using a series of spreadsheet .csv files and manipulating them 
with code. One point is that spreadsheets are somewhat easily altered by me/collaborators. 
I'd actually like to keep the data about all the files and their relationship somewhat 
difficult to alter without rigorous intent/thought. Having a formal database system 
to maintain all file records therefore is seeming more appealing at this point. 

As a Python based user I see there are three main options I have:
1. sqlite
1. mysql
1. postgresql

SQlite seems to be the quickest and lightest option amongst the three, though it is also the one that 
does not have multi-user access, or even server-type construction. The .db file is the central piece behind
the entire database, which makes it fast and light. MySQL is a popular option, though it is much heavier to 
install, and relies on a server - which slows down its operation speed. Postgresql is said to be somewhat 
bulky, very powerful, however to also have a steep learning curve. 

At this point, I'm leaning towards learning the required skills on sqlite, implementing a database, and 
then switching over the mysql at a later point once multiple users become regularly involved. Moreover, 
I also checked and there are GUI options to view .db files, which means I could also just send collaborators 
a .db file that they could examine the contents of the files independently without actually doing any coding. 

The option to export tables in .db files into .csv files is also always there irrespective of the actual 
framework in use - which also means I can send spreadsheets to collaborators as and when necessary. 


## What the 'end-goal' is
In the longer-term (+1 year from now), I'd like the dataset and the DBMS to 
be web-friendly. This would mean collaborators/users can access the dataset
in a user-friendly way and download the required raw and processed data at will. 

I would like the Ushichka dataset to remain a public dataset that will be the centre
for multi-disciplinary work between the applied and biological sciences. This means having
sufficient documentation on the data, its origins and the workflows data at 
various stages of processing have been through.

## Who are the users
The 'end-user' of this dataset will be primarily me in the early stages, followed by
my collaborators in the later stages, and in the final stages, the general scientific
community who may be interested in accessing the data to develop/investigate in
their own directions. 

# What are my requirements?

* Would like to rely on open-source software as much as possible, while avoiding proprietary software. 
Rather than a dislike for proprietary software, the motivation is to allow easy access of the data/DBMS to anyone interested in working with it, and 
avoiding issues like installation/licensing limitations. 
* Python based DBMS. The rest of my code and data processing workflows are in Python, and it would help me to integrate
the DBMS without switching paradigms too much. 
* The dataset/DBMS to be web accesible. Users should be able to interact with the DBMS +  access information about the dataset and 
also download data easily from anywhere in the world. 
 





