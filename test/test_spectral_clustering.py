import numpy as np
from manifolds.spectral_clustering import SpectralClustering
from visualise.visualise import GraphVisualiser

class Student:
    
    def __init__(self, exam_types):
        # assign random preferred exam type out of exam_types possible types
        self.preferred_type = np.random.randint(exam_types, size=1)[0]
    
    def sit_exams(self, exams, default_mean=65, pref_mean=75, default_std=10, pref_std=10):
        # compute the mean value of Gaussian distribution for exam grades for student
        mean_exams = np.zeros(shape=exams.shape)
        mean_exams[exams == self.preferred_type] = pref_mean
        mean_exams[exams != self.preferred_type] = default_mean
        
        # compute the std value of Gaussian distribution for student
        std_exams = np.zeros(shape=exams.shape)
        std_exams[exams == self.preferred_type] = pref_std
        std_exams[exams != self.preferred_type] = default_std
        
        # randomly sample grades from Gaussian distribution with mean and std
        exam_grades = np.random.normal(loc=mean_exams, scale=std_exams)
        
        # ensure exam results are between 0 and 100
        exam_grades[exam_grades > 100.0] = 100.0
        exam_grades[exam_grades < 0.0] = 0.0
        self.exam_grades = exam_grades
        
    def get_exam_grades(self):
        return self.exam_grades
    
    def get_type(self):
        return self.preferred_type


def create_exams(total_exams, types=3):
    # create array of exam types of length equal to the number of total exams
    return np.random.randint(types, size=total_exams)


def test_spectral_clustering():
    np.random.seed(21)
    total_exams = 40
    total_students = 70
    exam_types = 3
    exams = create_exams(total_exams, types=exam_types)
    
    # create a NxM matrix of exam results, where N is the number of students, and M is each exam.
    exam_results = []
    preferred_types = []
    for student in range(0, total_students):
        student = Student(exam_types)
        student.sit_exams(exams)
        exam_results.append(student.get_exam_grades())
        preferred_types.append(student.get_type())
        
    # create matrix
    X = np.vstack(exam_results)
    true_labels = np.array(preferred_types)

    sc = SpectralClustering(X, norm=True, kernel='rbf', gamma=1/10000, edge_thresh=90)
    w, v = sc.eig_decompose()
    edges = sc.edges

    # the first eigenvalue is a maximally smooth constant, can be omitted
    spectral_matrix = v[:, 1:]
    # create a reduced spectral vector, only use first 3 spectral components
    q = spectral_matrix[:, :3]

    # min-max normalise each component of spectral vector to range 0-1
    colors = (q - q.min(axis=0)) / (q.max(axis=0) - q.min(axis=0)) * 255
    colors = colors.astype(int)

    gv = GraphVisualiser(q, node_colors=true_labels, edges=edges, dimension='1d')
    fig = gv.visualise()
    fig.show()