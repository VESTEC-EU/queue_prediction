#Contains the database objects mapping 
import pony.orm as pny
import datetime 

#default name of the database
filename = "jobs.db"


db = pny.Database()

#For the Job table
class Job(db.Entity):
    uuid = pny.PrimaryKey(str)
    requested_nodes = pny.Required(float)
    requested_walltime = pny.Required(float)
    submit_time = pny.Required(datetime.datetime)
    start_time = pny.Required(datetime.datetime)
    finish_time = pny.Required(datetime.datetime)
    actual_walltime = pny.Required(float)
    machine = pny.Required(str)
    queue_name = pny.Required(str)
    actual_waittime = pny.Required(float)

    queued = pny.Set("Job",reverse="queued")
    running = pny.Set("Job",reverse="running")



def initialise_database(fname = None):
    if fname is None:
        db.bind(provider="sqlite",filename=filename,create_db=True)
    else:
        db.bind(provider="sqlite",filename=fname,create_db=True)
    db.generate_mapping(create_tables=True)


# if __name__ == "__main__":
#     initialise_database("test.db")